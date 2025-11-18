#!/bin/bash

cd ../vllm-project
pip install -r <(sed '/^[torch]/d' requirements/build.txt)
# pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
cd -

pip install -e .
