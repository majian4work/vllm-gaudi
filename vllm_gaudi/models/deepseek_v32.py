import os
import typing

import torch
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.layer import Attention
from vllm.config import CacheConfig, ParallelConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
import vllm.model_executor.models.deepseek_v2 as deepseek_v2
from vllm.model_executor.models.deepseek_v2 import Indexer, DeepseekV32IndexerCache, sparse_attn_indexer_fake
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder, CommonAttentionMetadata
import vllm.v1.worker.utils as utils
from vllm.v1.worker.utils import defaultdict, extract_layer_index


def per_token_group_quant_fp8_pytorch(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch-native implementation of per-token-group quantization for FP8.
    """
    if use_ue8m0 is None:
        # Default fallback - could import is_deep_gemm_e8m0_used if needed
        use_ue8m0 = False

    if dtype is None:
        dtype = current_platform.fp8_dtype()

    # Validate inputs
    assert x.shape[-1] % group_size == 0, (
        f"Last dimension {x.shape[-1]} must be divisible by group_size {group_size}"
    )
    assert x.stride(-1) == 1, "Input tensor groups must be contiguous"

    # Get FP8 range
    # finfo = torch.finfo(dtype)
    finfo = torch.finfo(torch.float8_e4m3fnuz) # HACK for gaudi2
    fp8_min = finfo.min
    fp8_max = finfo.max

    # Prepare output tensor
    if out_q is None:
        x_q = torch.empty_like(x, dtype=dtype)
    else:
        assert out_q.shape == x.shape
        x_q = out_q

    # Reshape input for group processing
    # Original shape: (..., last_dim)
    # Target shape: (..., num_groups, group_size)
    original_shape = x.shape
    num_groups = original_shape[-1] // group_size

    # Reshape to separate groups
    group_shape = original_shape[:-1] + (num_groups, group_size)
    x_grouped = x.view(group_shape)

    # Compute per-group absolute maximum values
    # Shape: (..., num_groups)
    abs_max = torch.amax(torch.abs(x_grouped), dim=-1, keepdim=False)
    abs_max = torch.maximum(abs_max, torch.tensor(eps, device=x.device, dtype=x.dtype))

    # Compute scales
    scale_raw = abs_max / fp8_max

    if use_ue8m0:
        # For UE8M0 format, scales must be powers of 2
        scales = torch.pow(2.0, torch.ceil(torch.log2(scale_raw)))
    else:
        scales = scale_raw

    # Expand scales for broadcasting with grouped data
    # Shape: (..., num_groups, 1)
    scales_expanded = scales.unsqueeze(-1)

    # Quantize the grouped data
    x_scaled = x_grouped / scales_expanded
    x_clamped = torch.clamp(x_scaled, fp8_min, fp8_max)
    x_quantized = x_clamped.to(dtype)

    # Reshape back to original shape
    x_q.copy_(x_quantized.view(original_shape))

    # Prepare scales tensor in requested format
    if column_major_scales:
        # Column-major: (num_groups,) + batch_dims
        # Transpose the scales to put group dimension first
        scales_shape = (num_groups,) + original_shape[:-1]
        x_s = scales.permute(-1, *range(len(original_shape) - 1))
        x_s = x_s.contiguous().view(scales_shape)
    else:
        # Row-major: batch_dims + (num_groups,)
        x_s = scales.contiguous()

    # Ensure scales are float32
    x_s = x_s.to(torch.float32)

    return x_q, x_s


def sparse_attn_indexer_pytorch(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of sparse_attn_indexer.

    This function performs sparse attention indexing by:
    1. Quantizing and caching K tensors
    2. Computing attention logits using FP8 operations
    3. Finding top-k attention indices for sparse attention
    """
    # Handle dummy run case
    attn_metadata = get_forward_context().attn_metadata
    # if not isinstance(attn_metadata, dict):
    #     return sparse_attn_indexer_fake(
    #         hidden_states,
    #         k_cache_prefix,
    #         kv_cache,
    #         q_fp8,
    #         k,
    #         weights,
    #         quant_block_size,
    #         scale_fmt,
    #         topk_tokens,
    #         head_dim,
    #         max_model_len,
    #         total_seq_lens,
    #         topk_indices_buffer,
    #     )

    # attn_metadata = attn_metadata[k_cache_prefix]
    # assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    is_prompt = attn_metadata.is_prompt

    # Initialize topk_indices_buffer
    # topk_indices_buffer[:q_fp8.shape[0], :] = -1
    # topk_indices_buffer[:hidden_states.shape[0]] = -1
    topk_indices_buffer.fill_(-1)

    k_fp8, k_scale = _pytorch_indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, scale_fmt
    )

    if is_prompt:
        logits = _pytorch_fp8_mqa_logits(
            q_fp8,
            k_fp8,
            k_scale,
            weights,
        )
        topk_indices = _pytorch_topk_with_bounds(logits, topk_tokens)
        topk_indices = topk_indices.view(-1, topk_indices.shape[-1])
        topk_indices_buffer[:topk_indices.shape[0], : topk_indices.shape[-1] ] = topk_indices.to(dtype=torch.int32)
    else:
        # PyTorch implementation of fp8_paged_mqa_logits
        logits = _pytorch_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache,
            weights,
            attn_metadata,
            max_model_len,
        )

        # Apply position masking and get top-k indices
        # q_fp8: [padded_token_num, num_heads, head_dim]
        padded_token_num = q_fp8.size(0)
        topk_indices = _pytorch_decode_topk_with_masking(
            logits,
            attn_metadata,
            topk_tokens, padded_token_num
        )
        topk_indices_buffer[:topk_indices.shape[0], : topk_indices.shape[-1]] = (
            topk_indices
        )

    return topk_indices_buffer


def _pytorch_indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
) -> None:
    """PyTorch implementation of indexer_k_quant_and_cache kernel.
    """
    head_dim = k.shape[-1]
    k = k.view(-1, head_dim)  # [total_tokens, head_dim]

    k_fp8, k_scale = per_token_group_quant_fp8_pytorch(
        k,
        group_size=quant_block_size,
        column_major_scales=False,
        use_ue8m0=(scale_fmt == "ue8m0"),
    )

    k_fp8_bytes = k_fp8.view(-1, head_dim).view(torch.uint8)
    scale_bytes = k_scale.view(torch.uint8).view(-1, 4)
    fp8_bytes = torch.cat([k_fp8_bytes, scale_bytes], dim=-1)  # [total_tokens, head_dim + 4]

    import habana_frameworks.torch.core as htcore
    htcore.mark_step()
    slot_mapping = slot_mapping.flatten()

    # from vllm_gaudi.extension.utils import VLLMKVCache
    # indexer_cache_k = VLLMKVCache()
    # indexer_cache_k(fp8_bytes, kv_cache, slot_mapping)
    # kv_cache: [num_block*block_size, head_dim + 4]
    kv_cache.index_copy_(0, slot_mapping, fp8_bytes)
    htcore.mark_step()

    return k_fp8, k_scale


def _pytorch_fp8_mqa_logits(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """PyTorch implementation of fp8_mqa_logits.

    Optimized with vectorized operations where possible.
    """

    # q: [padded_token_num, num_heads, head_dim] [.., 64, 128]
    # k: [padded_token_num, head_dim] [.., 128]
    # weights: [padded_token_num, num_heads] [.., 64]

    q_float = q_fp8.float()  # [padded_token_num, num_heads, head_dim]
    k_scale_f32 = k_scale.view(torch.float32)
    k_dequant = k_fp8.float() * k_scale_f32  # [padded_token_num, head_dim]

    logits = torch.matmul(q_float, k_dequant.T).relu()  # [padded_token_num, num_heads, padded_token_num]
    logits = logits * weights[..., None]
    logits = logits.sum(dim=1)  # [padded_token_num, padded_token_num]

    return logits


def _pytorch_topk_with_bounds(
    logits: torch.Tensor,
    topk_tokens: int,
) -> torch.Tensor:
    """PyTorch implementation of bounded top-k selection."""
    seq_len = logits.shape[0]

    mask = torch.triu(torch.ones(logits.shape, device=logits.device, dtype=torch.bool),
                        diagonal=1)

    # Apply bounds masking before topk
    logits = logits.masked_fill(mask, float('-inf'))

    # Get top-k indices
    topk_indices = logits.topk(min(topk_tokens, seq_len), dim=-1)[1]
    topk_indices = topk_indices.masked_fill(mask[:, :min(topk_tokens, seq_len)], -1)

    return topk_indices


def _pytorch_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    attn_metadata,
    max_model_len: int,
) -> torch.Tensor:
    """PyTorch implementation of fp8_paged_mqa_logits.

    Args:
        q_fp8: Query tensor of shape [B, H, D]. Casted to `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B, H], dtype `torch.float32`.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B, max_model_len], dtype
        `torch.float32`.
    """
    padded_token_num, q_heads, head_dim = q_fp8.shape
    # kv_cache: [num_blocks, block_size, D+4] -> [num_blocks, block_size, 1, D+4]
    kv_cache = kv_cache.unsqueeze(-2)  # Add head dimension
    kv_heads = kv_cache.shape[-2]

    block_list = attn_metadata.block_list
    block_mapping = attn_metadata.block_mapping

    from vllm_gaudi.extension.utils import VLLMKVCache
    from vllm_gaudi.extension.ops import batch2block

    q_float = q_fp8.float()
    q_float = batch2block(q_float, block_mapping.float()).unsqueeze(-2)
    # q_float: [padded_block_num, q_head, 1, head_dim] [:, 64, 1, 128]
    weights = batch2block(weights, block_mapping.float()).unsqueeze(-2)
    # weights: [padded_block_num, 1, q_head] [:, 1, 64]

    indexer_cache_k = VLLMKVCache()
    fetch_from_cache = indexer_cache_k.fetch_from_cache
    key = fetch_from_cache(kv_cache.unflatten(0, (-1, attn_metadata.block_size)), block_list)
    # key: [padded_block_num, block_size, kv_head, head_dim+4] [:, 128, 1, 132]

    k_fp8_val = key[..., :head_dim].view(torch.float8_e4m3fn)
    k_scale_val = key[..., head_dim:].view(torch.float32)
    k_dequant = k_fp8_val.float() * k_scale_val
    k_dequant = k_dequant.transpose(1, 2)

    if kv_heads != q_heads:
        assert q_heads % kv_heads == 0
        # q_fp8: [padded_block_num, q_head, 1, head_dim] [:, 64, 1, 128]
        # key: [padded_block_num, kv_head, block_size, head_dim] [:, 1, 128, 128]
        k_dequant = k_dequant.repeat_interleave(int(q_heads/kv_heads), dim=1)

    attn = torch.matmul(q_float, k_dequant.transpose(-2, -1)).squeeze(-2).relu()
    # attn: [padded_block_num, q_head, block_size] [:, 64, 128]
    attn = attn * weights.squeeze(-2)[..., None]
    attn = attn.sum(dim=1) # [padded_block_num, block_size] [:, 128]

    # Gather logits per sequence naively for verification, which don't support static shapes
    # logits = torch.zeros(padded_token_num, max_model_len,
    #                     device=q_fp8.device, dtype=torch.float32)
    # for token_idx in range(padded_token_num): # seq
    #     idx = block_groups == token_idx
    #     batch_block_indices = torch.nonzero(idx, as_tuple=False).squeeze(-1)
    #     selected = attn[batch_block_indices].flatten()
    #     logits[token_idx, :selected.shape[0]] = selected

    device = q_fp8.device
    num_blocks, block_size = attn.shape
    logits = torch.zeros(padded_token_num, num_blocks, block_size, dtype=attn.dtype, device=device)
    batch_block_mapping = attn_metadata.batch_block_mapping
    torch.gather(attn.unsqueeze(0), 0, batch_block_mapping, out=logits)
    return logits.view(padded_token_num, -1)


def _pytorch_decode_topk_with_masking(
    logits: torch.Tensor,
    attn_metadata,
    topk_tokens: int,
    padded_token_num: int,
) -> torch.Tensor:
    """PyTorch implementation of decode top-k with position masking."""
    current_device = logits.device

    len = logits.shape[1]
    # Create position mask
    positions = (
        torch.arange(len, device=current_device)
        .unsqueeze(0)
        .expand(padded_token_num, -1)
    )
    index_end_pos = attn_metadata.input_positions

    # Apply mask and get top-k
    mask = positions <= index_end_pos
    logits = logits.masked_fill(~mask, float("-inf"))
    topk_indices = logits.topk(topk_tokens, dim=-1)[1].to(torch.int32)

    # Clamp out-of-range indices
    topk_indices[topk_indices > index_end_pos] = -1

    return topk_indices


def hpu_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    # Check if PyTorch implementation should be used
    return sparse_attn_indexer_pytorch(
        hidden_states, k_cache_prefix, kv_cache, q_fp8, k, weights,
        quant_block_size, scale_fmt, topk_tokens, head_dim, max_model_len,
        total_seq_lens, topk_indices_buffer
    )


direct_register_custom_op(
    op_name="hpu_sparse_attn_indexer",
    op_func=hpu_sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


# The original Builder has some code related to cuda sm.
class HPUDeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # We don't use build() from this class
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ):
        pass


class HPUDeepseekV32IndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_builder_cls() -> type["HPUDeepseekV32IndexerMetadataBuilder"]:
        return HPUDeepseekV32IndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks*block_size, head_size)


class HPUDeepseekV32IndexerCache(DeepseekV32IndexerCache):
    def get_attn_backend(self) -> AttentionBackend:
        return HPUDeepseekV32IndexerBackend


class HPUIndexer(Indexer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__(vllm_config, config, hidden_size, q_lora_rank, quant_config, cache_config, topk_indices_buffer, prefix)

        # remove already register cache layer in Indexer.
        compilation_config = get_current_vllm_config().compilation_config
        del compilation_config.static_forward_context[f"{prefix}.k_cache"]
        self.k_cache = HPUDeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )

    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
    ) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        k, _ = self.wk(hidden_states)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe.squeeze(1), k_nope], dim=-1)

        # we only quant q here since k quant is fused with cache insertion
        q = q.view(-1, self.head_dim)

        q_fp8, q_scale = per_token_group_quant_fp8_pytorch(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )

        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim) # [padded_token_num, n_head, head_dim] n_head: 64
        q_scale = q_scale.view(-1, self.n_head, 1) # [padded_token_num, n_head, 1]

        weights, _ = self.weights_proj(hidden_states) # [padded_token_num, n_head]
        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
        )
        weights = weights.squeeze(-1)
        return torch.ops.vllm.hpu_sparse_attn_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache[0][0],
            q_fp8,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )


def hpu_bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, "Attention"],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int | None = 1,
) -> None:
    """
    Bind the allocated KV cache to both ModelRunner and forward context so
    that the KV cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's kv cache list (`runner_kv_caches`) with
         kv_caches.
      2) Associates each attention layer in the `forward_context` with its
         corresponding KV cache in kv_caches.

    Args:
        kv_caches: The allocated kv_caches with layer names as keys.
        forward_context: The global forward context containing all Attention
            layers with layer names as keys.
        runner_kv_caches: The kv_cache declared by ModelRunner.
    """
    # Bind kv_caches to ModelRunner
    assert len(runner_kv_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.

            # TODO - analyze where runner_kv_caches is used and the right
            # way to ensure it properly reflects multiple attention layers
            # in the same decoder block.
            if current_platform.is_cuda() or current_platform.is_xpu():
                # We know that the GPU runner is not impacted by this
                # case. Some test code depends on runner_kv_caches, but
                # not in a way that's impacted by ignoring this.
                pass
            else:
                # HACK: for hpu
                pass
                # raise NotImplementedError
        layer_name = layer_names[0]
        runner_kv_caches.append(kv_caches[layer_name])

    # Bind kv_caches to forward context
    for layer_name, kv_cache in kv_caches.items():
        # NOTE: Use list because of v0 PP virtual engine.
        forward_context[layer_name].kv_cache = [kv_cache]

deepseek_v2.Indexer = HPUIndexer
utils.bind_kv_cache = hpu_bind_kv_cache