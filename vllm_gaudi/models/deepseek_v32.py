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

from vllm_gaudi.extension.ops import FP8_MAX

def _pytorch_group_quant(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    scale_raw = abs_max / FP8_MAX

    if use_ue8m0:
        # For UE8M0 format, scales must be powers of 2
        scales = torch.pow(2.0, torch.ceil(torch.log2(scale_raw)))
    else:
        scales = scale_raw

    # Expand scales for broadcasting with grouped data
    # Shape: (..., num_groups, 1)
    scales_expanded = scales.unsqueeze(-1)

    # Quantize the grouped data
    x_quantized = torch.ops.hpu.cast_to_fp8_v2(x_grouped, 1.0 / scales_expanded, False, False, torch.float8_e4m3fn)[0]

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
    return x_q, x_s.float()


def _pytorch_indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
) -> None:
    head_dim = k.shape[-1]
    k = k.view(-1, head_dim)  # [total_tokens, head_dim]

    if os.environ.get("VLLM_INDEXER_FP8_CACHE", "0") in ["1", "true", "True"]:
        k_fp8, k_scale = _pytorch_group_quant(
            k,
            group_size=quant_block_size,
            column_major_scales=False,
            use_ue8m0=(scale_fmt == "ue8m0"),
        )

        k_fp8_bytes = k_fp8.view(-1, head_dim).view(torch.uint8)
        scale_bytes = k_scale.view(torch.uint8).view(-1, 4)
        k = torch.cat([k_fp8_bytes, scale_bytes], dim=-1)  # [total_tokens, head_dim + 4]

    slot_mapping = slot_mapping.flatten()
    # kv_cache: [num_block*block_size, head_dim + 4]
    kv_cache.index_copy_(0, slot_mapping, k)

    # from vllm_gaudi.extension.utils import VLLMKVCache
    # indexer_cache_k = VLLMKVCache()
    # indexer_cache_k(k, kv_cache, slot_mapping)


def _pytorch_fp8_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    # q: [padded_token_num, num_heads, head_dim] [.., 64, 128]
    # k: [padded_token_num, head_dim] [.., 128]
    # weights: [padded_token_num, num_heads] [.., 64]
    logits = torch.matmul(q, k.T).relu()  # [padded_token_num, num_heads, padded_token_num]
    logits = logits * weights[..., None]
    logits = logits.sum(dim=1)  # [padded_token_num, padded_token_num]

    return logits


def _pytorch_topk_with_bounds(
    logits: torch.Tensor,
    topk_tokens: int,
) -> torch.Tensor:
    device = logits.device
    seq_len = logits.shape[0]

    # Caution: Only support batch_size=1 w/o chunked-prefill
    mask = torch.triu(torch.ones(logits.shape, device=device, dtype=torch.bool), diagonal=1)

    # Apply bounds masking before topk
    logits = logits.masked_fill(mask, float('-inf'))

    # Get top-k indices
    topk_indices = logits.topk(min(topk_tokens, seq_len), dim=-1)[1].to(torch.int32)
    # Clamp out-of-range indices
    topk_indices = topk_indices.masked_fill(mask[:, :min(topk_tokens, seq_len)], -1)

    # flatten batch_size*padded_seq_len if batch_size>1
    # topk_indices = topk_indices.view(-1, topk_indices.shape[-1])

    return topk_indices


def _pytorch_fp8_paged_mqa_logits(
    # q_fp8: torch.Tensor,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    attn_metadata,
) -> torch.Tensor:
    # padded_token_num, q_heads, head_dim = q_fp8.shape
    padded_token_num, q_heads, head_dim = q.shape
    # kv_cache: [num_blocks, block_size, D+4] -> [num_blocks, block_size, 1, D+4]
    kv_cache = kv_cache.unsqueeze(-2)  # Add head dimension
    kv_heads = kv_cache.shape[-2]

    block_list = attn_metadata.block_list
    block_mapping = attn_metadata.block_mapping

    from vllm_gaudi.extension.utils import VLLMKVCache
    from vllm_gaudi.extension.ops import batch2block

    # q = q_fp8.float()
    # q = batch2block(q, block_mapping.float()).unsqueeze(-2)
    q = batch2block(q, block_mapping).unsqueeze(-2)
    # q: [padded_block_num, q_head, 1, head_dim] [:, 64, 1, 128]
    # weights = batch2block(weights, block_mapping.float()).unsqueeze(-2)
    weights = batch2block(weights, block_mapping).unsqueeze(-2)
    # weights: [padded_block_num, 1, q_head] [:, 1, 64]

    indexer_cache_k = VLLMKVCache()
    fetch_from_cache = indexer_cache_k.fetch_from_cache
    key = fetch_from_cache(kv_cache.unflatten(0, (-1, attn_metadata.block_size)), block_list)
    # key: [padded_block_num, block_size, kv_head, head_dim+4] [:, 128, 1, 132]

    if os.environ.get("VLLM_INDEXER_FP8_CACHE", "0") in ["1", "true", "True"]:
        k_fp8_val = key[..., :head_dim].view(torch.float8_e4m3fn)
        k_scale_val = key[..., head_dim:].view(torch.float32)
        k_dequant = k_fp8_val.float() * k_scale_val
        key = k_dequant.to(torch.bfloat16)
    key = key.transpose(1, 2)

    if kv_heads != q_heads:
        assert q_heads % kv_heads == 0
        # q_fp8: [padded_block_num, q_head, 1, head_dim] [:, 64, 1, 128]
        # key: [padded_block_num, kv_head, block_size, head_dim] [:, 1, 128, 128]
        key = key.repeat_interleave(int(q_heads/kv_heads), dim=1)

    # attn = torch.matmul(q_float, k_dequant.transpose(-2, -1)).squeeze(-2).relu()
    attn = torch.matmul(q, key.transpose(-2, -1)).squeeze(-2).relu()
    # attn: [padded_block_num, q_head, block_size] [:, 64, 128]
    attn = attn * weights.squeeze(-2)[..., None]
    attn = attn.sum(dim=1) # [padded_block_num, block_size] [:, 128]

    device = q.device
    num_blocks, block_size = attn.shape
    logits = torch.zeros(padded_token_num, num_blocks, block_size, dtype=attn.dtype, device=device)
    batch_block_mapping = attn_metadata.batch_block_mapping
    torch.gather(attn.unsqueeze(0), 0, batch_block_mapping, out=logits)
    return logits.view(padded_token_num, -1)


def _pytorch_decode_topk_with_masking(
    logits: torch.Tensor,
    topk_tokens: int,
    max_model_len: int,
    attn_metadata,
) -> torch.Tensor:
    device = logits.device
    padded_token_num, flatten_key_len = logits.shape
    key_len = min(max_model_len, flatten_key_len)
    logits = logits[:, :key_len]

    # Create position mask
    positions = (
        torch.arange(key_len, device=device)
        .unsqueeze(0)
        .expand(padded_token_num, -1)
    )
    index_end_pos = attn_metadata.input_positions
    mask = positions <= index_end_pos

    # Apply masking before topk
    logits = logits.masked_fill(~mask, float("-inf"))

    # Get top-k indices
    topk_indices = logits.topk(min(key_len, topk_tokens), dim=-1)[1].to(torch.int32)
    # Clamp out-of-range indices
    # print(f"topk_indices {topk_indices}, index_end_pos {index_end_pos}")
    topk_indices[topk_indices > index_end_pos] = -1
    # print(f"topk_indices after clamp {topk_indices}")

    return topk_indices


def hpu_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
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
    # Handle dummy run case
    attn_metadata = get_forward_context().attn_metadata
    # if not isinstance(attn_metadata, dict):
    #     return sparse_attn_indexer_fake(
    #         hidden_states,
    #         k_cache_prefix,
    #         kv_cache,
    #         q,
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

    _pytorch_indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, scale_fmt
    )

    if is_prompt:
        logits = _pytorch_fp8_mqa_logits(
            q,
            k,
            weights,
        )
        topk_indices = _pytorch_topk_with_bounds(logits, topk_tokens)
    else:
        logits = _pytorch_fp8_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            attn_metadata,
        )
        topk_indices = _pytorch_decode_topk_with_masking(
            logits,
            topk_tokens,
            max_model_len,
            attn_metadata,
        )

    # Initialize topk_indices_buffer
    import habana_frameworks.torch.core as htcore
    htcore.mark_step()
    # topk_indices_buffer[:topk_indices.shape[0]] = -1
    topk_indices_buffer.fill_(-1)
    htcore.mark_step()
    topk_indices_buffer[:topk_indices.shape[0], :topk_indices.shape[-1]] = topk_indices
    # topk_indices_buffer[:topk_indices.shape[0], topk_indices.shape[-1]:] = -1
    return topk_indices_buffer


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
        if os.environ.get("VLLM_INDEXER_FP8_CACHE", "0") in ["1", "true", "True"]:
            dtype = torch.uint8
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
        else:
            dtype = torch.get_default_dtype()
            head_dim=self.head_dim
        self.k_cache = HPUDeepseekV32IndexerCache(
            head_dim=head_dim,
            dtype=dtype,
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

        weights, _ = self.weights_proj(hidden_states) # [padded_token_num, n_head]
        weights = weights * self.softmax_scale * self.n_head**-0.5
        return torch.ops.vllm.hpu_sparse_attn_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache[0][0],
            q,
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

            # HACK: for hpu
            pass
        layer_name = layer_names[0]
        runner_kv_caches.append(kv_caches[layer_name])

    # Bind kv_caches to forward context
    for layer_name, kv_cache in kv_caches.items():
        # NOTE: Use list because of v0 PP virtual engine.
        forward_context[layer_name].kv_cache = [kv_cache]

deepseek_v2.Indexer = HPUIndexer
utils.bind_kv_cache = hpu_bind_kv_cache