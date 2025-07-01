import torch
from gemm_kernels import gemm_tn_test

def measure_error_stats(pred, label):
    abs_diff = torch.abs(pred - label)
    max_err = abs_diff.max().item()
    avg_err = abs_diff.mean().item()
    print(pred)
    print(label)
    assert torch.allclose(pred, label, atol=1e-2, rtol=1e-2), f"Outputs are not close enough: max_err={max_err}, avg_err={avg_err}"
    return max_err, avg_err

if __name__ == "__main__":
    # Define matrix shape and data type
    shapes = [(4096, 4096), (2048, 2048), (1024, 1024)]
    for shape in shapes:
        dtype = torch.bfloat16
        device = 'cuda'
        
        print(f"Testing GEMM for shape {shape} with dtype {dtype} on {device}")
        
        # 1. Create input tensors
        A = torch.randn(shape, device=device, dtype=dtype)
        B = torch.randn(shape, device=device, dtype=dtype)
        
        # 2. Create SEPARATE output tensors for each implementation
        C_custom = torch.empty((shape[0], shape[0]), device=device, dtype=dtype)
        C_custom.fill_(float("NaN"))  # Initialize with NaN to detect uninitialized values
        C_reference = torch.empty((shape[0], shape[0]), device=device, dtype=torch.float32)

        # 3. Run the custom kernel
        # Assuming gemm_tn_test modifies C_custom in-place
        gemm_tn_test(A, B, C_custom)
        #torch.matmul(A, B.T, out=C_custom)
        # 4. Run the reference implementation (this is the ground truth)
        torch.matmul(A.float(), B.float().T, out=C_reference)
        
        # 5. Compare the results using the robust verification function
        print("Comparing results...")
        max_err, avg_err = measure_error_stats(C_custom.float(), C_reference)
        print(f"Max error: {max_err}, Avg error: {avg_err}")