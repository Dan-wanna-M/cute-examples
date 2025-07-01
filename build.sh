export CPLUS_INCLUDE_PATH="$PWD:$PWD/libs/cutlass/include:$CPLUS_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="/workspace/cute-examples/.venv/lib/python3.12/site-packages/torch/include:$CPLUS_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="/workspace/cute-examples/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include/:$CPLUS_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="/usr/local/cuda/include:$CPLUS_INCLUDE_PATH"
export CXX="g++"
export TORCH_CUDA_ARCH_LIST="9.0a"
pip install ./kernels -v --no-build-isolation