import torch
from gemm_kernels import gemm_tn_test 
shape = (4096, 4096)  # Example shape
dtype = torch.bfloat16  # Example data type
device = 'cuda'  # Example device
A = torch.randn(shape, device=device, dtype=dtype)
B = torch.randn(shape, device=device, dtype=dtype)

# 2. Create SEPARATE output tensors for each implementation
C_custom = torch.empty((shape[0], shape[0]), device=device, dtype=dtype)

gemm_tn_test(A, B, C_custom)