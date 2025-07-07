import torch
import time
import numpy as np
import ctypes
from ctypes import c_void_p, c_int, c_float, byref, POINTER

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda import gpuarray
    import skcuda.cublas as cublas
    CUBLAS_AVAILABLE = True
except ImportError:
    print("Warning: PyCUDA/scikit-cuda not available. Install with: pip install pycuda scikit-cuda")
    CUBLAS_AVAILABLE = False

from gemm_kernels import gemm_tn_test

# cuBLAS constants for bfloat16
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUDA_R_16BF = 14  # bfloat16 data type
CUDA_R_32F = 0    # float32 for compute type
CUBLAS_GEMM_DEFAULT = -1

# Additional cuBLAS algorithms to test
CUBLAS_ALGORITHMS = [
    -1,   # CUBLAS_GEMM_DEFAULT
    0,    # CUBLAS_GEMM_ALGO0
    1,    # CUBLAS_GEMM_ALGO1
    2,    # CUBLAS_GEMM_ALGO2
    3,    # CUBLAS_GEMM_ALGO3
    4,    # CUBLAS_GEMM_ALGO4
    5,    # CUBLAS_GEMM_ALGO5
    6,    # CUBLAS_GEMM_ALGO6
    7,    # CUBLAS_GEMM_ALGO7
    99,   # CUBLAS_GEMM_DFALT_TENSOR_OP
    100,  # CUBLAS_GEMM_ALGO0_TENSOR_OP
    101,  # CUBLAS_GEMM_ALGO1_TENSOR_OP
    102,  # CUBLAS_GEMM_ALGO2_TENSOR_OP
    103,  # CUBLAS_GEMM_ALGO3_TENSOR_OP
    104,  # CUBLAS_GEMM_ALGO4_TENSOR_OP
]

# Global cache for autotuned algorithms
_algorithm_cache = {}

def torch_bf16_to_cuda_bf16(tensor):
    """Convert PyTorch bfloat16 tensor to format compatible with cuBLAS."""
    # PyTorch bfloat16 and CUDA bfloat16 should be compatible
    return tensor.detach().cpu().numpy().view(np.uint16)

def create_test_tensors(shape, device='cuda', dtype=torch.bfloat16):
    """Create new test tensors for benchmarking."""
    A = torch.randn(shape[0], shape[1], device=device, dtype=dtype)
    B = torch.randn(shape[1], shape[0], device=device, dtype=dtype)
    C = torch.empty((A.shape[0], B.shape[1]), device=device, dtype=dtype)
    bytes_per_element = torch.finfo(dtype).bits // 8
    l2_flush_size_elements = (80 * 1024 * 1024) // bytes_per_element
    D = torch.randn(l2_flush_size_elements, dtype=dtype, device=device)
    D.mul_(1.01)
    return A, B, C

def benchmark_torch_native_gemm(shape, num_runs=100, device='cuda', dtype=torch.bfloat16, operation='mm'):
    """
    Benchmark PyTorch native GEMM operations with new tensors each run.
    
    Args:
        shape: Matrix dimensions (M, K)
        num_runs: Number of benchmark runs
        device: CUDA device
        dtype: Data type
        operation: Type of operation ('mm', 'addmm', 'bmm', 'baddbmm')
    
    Returns:
        Average time in seconds
    """
    times = []
    
    for _ in range(num_runs):
        # Create new tensors for each run
        A, B, C = create_test_tensors(shape, device, dtype)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        if operation == 'mm':
            # Standard matrix multiplication: A @ B.T
            torch.mm(A, B, out=C)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
    
    return np.mean(times)

def benchmark_torch_native_variants(shape, num_runs=100, device='cuda', dtype=torch.bfloat16):
    """
    Benchmark different PyTorch native GEMM variants.
    
    Returns:
        Dictionary with results for each variant
    """
    print(f"Benchmarking PyTorch native GEMM variants...")
    
    variants = ['mm']
    results = {}
    
    for variant in variants:
        try:
            print(f"  Testing torch.{variant}...")
            avg_time = benchmark_torch_native_gemm(shape, num_runs, device, dtype, variant)
            results[variant] = avg_time
            print(f"    Average time: {avg_time*1000:.4f} ms")
        except Exception as e:
            print(f"    Failed: {e}")
            results[variant] = None
    
    return results

def autotune_cublas_algorithm(shape, handle, device='cuda', dtype=torch.bfloat16, num_trials=5):
    """
    Autotune cuBLAS algorithm selection for optimal performance.
    
    Args:
        shape: Tuple of (M, K) dimensions
        handle: cuBLAS handle
        device: CUDA device
        dtype: Data type (bfloat16)
        num_trials: Number of trials per algorithm
    
    Returns:
        best_algorithm: Algorithm ID that performed best
        best_time: Best time achieved
    """
    cache_key = (shape, str(dtype), device)
    if cache_key in _algorithm_cache:
        print(f"Using cached algorithm {_algorithm_cache[cache_key]} for shape {shape}")
        return _algorithm_cache[cache_key], None
    
    print(f"Autotuning cuBLAS algorithms for shape {shape}...")
    
    # Get tensor dimensions
    m, k = shape[0], shape[1]
    n = shape[0]  # B is (k, n)
    
    # Scalar values
    alpha_val = np.array([1.0], dtype=np.float32)
    beta_val = np.array([0.0], dtype=np.float32)
    
    # Load cuBLAS library
    libcublas = ctypes.CDLL('/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcublas.so.12.6.4.1')
    

    if libcublas is None:
        print("Could not load cuBLAS library for autotuning")
        return CUBLAS_GEMM_DEFAULT, None
    
    # Setup function signatures
    libcublas.cublasGemmEx.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_int,     # transa
        ctypes.c_int,     # transb  
        ctypes.c_int,     # m
        ctypes.c_int,     # n
        ctypes.c_int,     # k
        ctypes.c_void_p,  # alpha
        ctypes.c_void_p,  # A
        ctypes.c_int,     # Atype
        ctypes.c_int,     # lda
        ctypes.c_void_p,  # B
        ctypes.c_int,     # Btype  
        ctypes.c_int,     # ldb
        ctypes.c_void_p,  # beta
        ctypes.c_void_p,  # C
        ctypes.c_int,     # Ctype
        ctypes.c_int,     # ldc
        ctypes.c_int,     # computeType
        ctypes.c_int      # algo
    ]
    libcublas.cublasGemmEx.restype = ctypes.c_int
    
    best_algorithm = CUBLAS_GEMM_DEFAULT
    best_time = float('inf')
    algorithm_results = {}
    
    # Test each algorithm
    for algo in CUBLAS_ALGORITHMS:
        times = []
        failed = False
        
        for trial in range(num_trials):
            # Create fresh tensors for each trial
            A_test, B_test, C_test = create_test_tensors(shape, device, dtype)
            A_ptr = A_test.data_ptr()
            B_ptr = B_test.data_ptr()
            C_ptr = C_test.data_ptr()
            
            # Warmup
            if trial == 0:
                torch.cuda.synchronize()
                status = libcublas.cublasGemmEx(
                    handle.value,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    n, m, k,
                    alpha_val.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_void_p(B_ptr), CUDA_R_16BF, n,
                    ctypes.c_void_p(A_ptr), CUDA_R_16BF, k,
                    beta_val.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_void_p(C_ptr), CUDA_R_16BF, n,
                    CUDA_R_32F,
                    algo
                )
                torch.cuda.synchronize()
                
                if status != 0:
                    print(f"Algorithm {algo} failed during warmup (status {status})")
                    failed = True
                    break
            
            # Actual timing
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            status = libcublas.cublasGemmEx(
                handle.value,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k,
                alpha_val.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_void_p(B_ptr), CUDA_R_16BF, n,
                ctypes.c_void_p(A_ptr), CUDA_R_16BF, k,
                beta_val.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_void_p(C_ptr), CUDA_R_16BF, n,
                CUDA_R_32F,
                algo
            )
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            if status != 0:
                print(f"Algorithm {algo} failed during trial {trial} (status {status})")
                failed = True
                break
            
            times.append(end_time - start_time)
        
        if not failed and times:
            avg_time = np.mean(times)
            min_time = np.min(times)
            std_time = np.std(times)
            
            algorithm_results[algo] = {
                'avg_time': avg_time,
                'min_time': min_time,
                'std_time': std_time,
                'times': times
            }
            
            print(f"  Algorithm {algo:3d}: {avg_time*1000:.3f}ms avg, {min_time*1000:.3f}ms best, {std_time*1000:.3f}ms std")
            
            # Update best algorithm based on average time
            if avg_time < best_time:
                best_time = avg_time
                best_algorithm = algo
        else:
            print(f"  Algorithm {algo:3d}: FAILED")
    
    # Cache the result
    _algorithm_cache[cache_key] = best_algorithm
    
    print(f"Best algorithm: {best_algorithm} ({best_time*1000:.3f}ms)")
    return best_algorithm, best_time

def benchmark_gemm_custom(shape, num_runs=100, device='cuda', dtype=torch.bfloat16):
    """Benchmark the custom GEMM function with new tensors each run."""
    times = []
    for _ in range(num_runs):
        # Create new tensors for each run
        A, B, C = create_test_tensors(shape, device, dtype)
        B = B.t().contiguous()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        result = gemm_tn_test(A, B, C)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
    
    return np.mean(times)

def benchmark_cublas_bf16_gemm_autotuned(shape, num_runs=100, device='cuda', dtype=torch.bfloat16, autotune=True):
    """
    Enhanced cuBLAS GEMM with autotuning for algorithm selection.
    
    Args:
        shape: Matrix dimensions (M, K)
        num_runs: Number of benchmark runs
        device: CUDA device
        dtype: Data type
        autotune: Whether to perform autotuning
    """
    if not CUBLAS_AVAILABLE:
        return None
    
    try:
        # Load cuBLAS library
        libcublas = None
        for lib_name in ['/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcublas.so.12.6.4.1']:
            try:
                libcublas = ctypes.CDLL(lib_name)
                print(f"Using cuBLAS library: {lib_name}")
                break
            except OSError:
                continue
        
        if libcublas is None:
            print("Could not load cuBLAS library")
            return None
        
        # Function signatures
        libcublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        libcublas.cublasCreate_v2.restype = ctypes.c_int
        
        libcublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
        libcublas.cublasDestroy_v2.restype = ctypes.c_int
        
        libcublas.cublasGemmEx.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_int,     # transa
            ctypes.c_int,     # transb  
            ctypes.c_int,     # m
            ctypes.c_int,     # n
            ctypes.c_int,     # k
            ctypes.c_void_p,  # alpha
            ctypes.c_void_p,  # A
            ctypes.c_int,     # Atype
            ctypes.c_int,     # lda
            ctypes.c_void_p,  # B
            ctypes.c_int,     # Btype  
            ctypes.c_int,     # ldb
            ctypes.c_void_p,  # beta
            ctypes.c_void_p,  # C
            ctypes.c_int,     # Ctype
            ctypes.c_int,     # ldc
            ctypes.c_int,     # computeType
            ctypes.c_int      # algo
        ]
        libcublas.cublasGemmEx.restype = ctypes.c_int
        
        # Create cuBLAS handle
        handle = ctypes.c_void_p()
        status = libcublas.cublasCreate_v2(byref(handle))
        if status != 0:
            print(f"cuBLAS create failed with status {status}")
            return None
        
        # Get tensor dimensions
        m, k = shape[0], shape[1]
        n = shape[0]  # B is (k, n)
        
        # Scalar values
        alpha_val = np.array([1.0], dtype=np.float32)
        beta_val = np.array([0.0], dtype=np.float32)
        
        # Perform autotuning if requested
        if autotune:
            best_algorithm, _ = autotune_cublas_algorithm(shape, handle, device, dtype)
        else:
            best_algorithm = CUBLAS_GEMM_DEFAULT
            print(f"Using default algorithm: {best_algorithm}")
        
        # Benchmark with the best algorithm
        print(f"Benchmarking with algorithm {best_algorithm}...")
        times = []
        
        for _ in range(num_runs):
            # Create new tensors for each run
            A_torch, B_torch, C_torch = create_test_tensors(shape, device, dtype)
            A_ptr = A_torch.data_ptr()
            B_ptr = B_torch.data_ptr() 
            C_ptr = C_torch.data_ptr()
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            status = libcublas.cublasGemmEx(
                handle.value,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k,
                alpha_val.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_void_p(B_ptr), CUDA_R_16BF, n,
                ctypes.c_void_p(A_ptr), CUDA_R_16BF, k,
                beta_val.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_void_p(C_ptr), CUDA_R_16BF, n,
                CUDA_R_32F,
                best_algorithm
            )
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            if status != 0:
                print(f"cuBLAS gemm failed with status {status}")
                break
            
            times.append(end_time - start_time)
        
        # Cleanup
        libcublas.cublasDestroy_v2(handle)
        
        if status == 0 and times:
            return np.mean(times)
        else:
            return None
            
    except Exception as e:
        print(f"cuBLAS autotuned benchmark failed: {e}")
        return None

def benchmark_cublas_lt_bf16(shape, num_runs=100, device='cuda', dtype=torch.bfloat16):
    """Alternative: Use cublasLt for bfloat16 GEMM with tensor cores and new tensors each run."""
    try:
        times = []
        
        for _ in range(num_runs):
            # Create new tensors for each run
            A_test, B_test, _ = create_test_tensors(shape, device, dtype)
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            _ = torch.mm(A_test, B_test)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        return np.mean(times)
        
    except Exception as e:
        print(f"cuBLAS LT benchmark failed: {e}")
        return None

def benchmark_comparison(shape=(4096, 4096), num_warmup=10, num_runs=100, autotune_cublas=True):
    """
    Compare custom GEMM with PyTorch native and autotuned cuBLAS bfloat16 operations.
    
    Args:
        shape: Matrix dimensions
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        autotune_cublas: Whether to autotune cuBLAS algorithms
    """
    device = 'cuda'
    dtype = torch.bfloat16
    
    print(f"Benchmarking BF16 GEMM with shape: {shape}")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Warmup runs: {num_warmup}, Benchmark runs: {num_runs}")
    print(f"cuBLAS autotuning: {'enabled' if autotune_cublas else 'disabled'}")
    print("Note: Creating new tensors for each run")
    print("-" * 80)
    
    # Warmup all implementations
    print("Warming up all implementations...")
    for _ in range(num_warmup):
        A, B, C = create_test_tensors(shape, device, dtype)
        # Warmup custom kernel
        result = gemm_tn_test(A, B.t().contiguous(), C)
        
        # Warmup PyTorch native
        result_torch = torch.mm(A, B)
        
        torch.cuda.synchronize()
    
    # Benchmark custom kernel
    print("Benchmarking custom GEMM...")
    custom_time = benchmark_gemm_custom(shape, num_runs, device, dtype)
    
    # Benchmark PyTorch native GEMM variants
    print("\nBenchmarking PyTorch native GEMM...")
    torch_results = benchmark_torch_native_variants(shape, num_runs, device, dtype)
    
    # Get the best PyTorch native result
    torch_best_time = None
    torch_best_variant = None
    for variant, time_val in torch_results.items():
        if time_val is not None:
            if torch_best_time is None or time_val < torch_best_time:
                torch_best_time = time_val
                torch_best_variant = variant
    
    # Calculate metrics
    M, N, K = shape[0], shape[0], shape[1]
    flops_per_op = 2 * M * N * K
    
    # Custom GEMM results
    custom_tflops = (flops_per_op / custom_time) / 1e12
    custom_time_ms = custom_time * 1000
    
    print(f"\nCustom GEMM Results:")
    print(f"  Average time: {custom_time_ms:.4f} ms")
    print(f"  Performance: {custom_tflops:.2f} TFLOPS")
    
    # PyTorch native results
    if torch_best_time is not None:
        torch_tflops = (flops_per_op / torch_best_time) / 1e12
        torch_time_ms = torch_best_time * 1000
        
        print(f"\nPyTorch Native GEMM Results (best: {torch_best_variant}):")
        print(f"  Average time: {torch_time_ms:.4f} ms")
        print(f"  Performance: {torch_tflops:.2f} TFLOPS")
    
    # cuBLAS comparison with autotuning
    print(f"\nBenchmarking cuBLAS BF16 {'(with autotuning)' if autotune_cublas else '(default algorithm)'}...")
    cublas_time = benchmark_cublas_bf16_gemm_autotuned(shape, num_runs, device, dtype, autotune_cublas)
    
    if cublas_time is None:
        print("Autotuned cuBLAS API failed, trying PyTorch backend...")
        cublas_time = benchmark_cublas_lt_bf16(shape, num_runs, device, dtype)
    
    results = {
        'custom_time_ms': custom_time_ms,
        'custom_tflops': custom_tflops,
        'torch_time_ms': torch_time_ms if torch_best_time else None,
        'torch_tflops': torch_tflops if torch_best_time else None,
        'torch_best_variant': torch_best_variant,
        'torch_all_results': torch_results,
        'cublas_time_ms': None,
        'cublas_tflops': None,
        'speedup_vs_torch': None,
        'speedup_vs_cublas': None
    }
    
    if cublas_time is not None:
        cublas_tflops = (flops_per_op / cublas_time) / 1e12
        cublas_time_ms = cublas_time * 1000
        
        print(f"cuBLAS BF16 Results:")
        print(f"  Average time: {cublas_time_ms:.4f} ms")
        print(f"  Performance: {cublas_tflops:.2f} TFLOPS")
        
        results['cublas_time_ms'] = cublas_time_ms
        results['cublas_tflops'] = cublas_tflops
        results['speedup_vs_cublas'] = cublas_time / custom_time
    
    # Performance comparison
    print(f"\nPerformance Comparison:")
    print(f"  Custom GEMM (BF16):     {custom_tflops:.2f} TFLOPS")
    
    if torch_best_time is not None:
        speedup_vs_torch = torch_best_time / custom_time
        results['speedup_vs_torch'] = speedup_vs_torch
        print(f"  PyTorch Native (BF16):  {torch_tflops:.2f} TFLOPS")
        print(f"  Speedup vs PyTorch:     {speedup_vs_torch:.2f}x {'(faster)' if speedup_vs_torch > 1 else '(slower)'}")
    
    if cublas_time is not None:
        speedup_vs_cublas = cublas_time / custom_time
        print(f"  cuBLAS (BF16):          {cublas_tflops:.2f} TFLOPS")
        print(f"  Speedup vs cuBLAS:      {speedup_vs_cublas:.2f}x {'(faster)' if speedup_vs_cublas > 1 else '(slower)'}")
        
    # Efficiency analysis
    theoretical_peak = get_theoretical_peak_tflops()
    if theoretical_peak:
        custom_efficiency = (custom_tflops / theoretical_peak) * 100
        torch_efficiency = (torch_tflops / theoretical_peak) * 100 if torch_best_time else None
        cublas_efficiency = (cublas_tflops / theoretical_peak) * 100 if cublas_time else None
        
        print(f"\nEfficiency Analysis:")
        print(f"  Theoretical Peak: {theoretical_peak:.1f} TFLOPS")
        print(f"  Custom efficiency: {custom_efficiency:.1f}%")
        if torch_efficiency:
            print(f"  PyTorch efficiency: {torch_efficiency:.1f}%")
        if cublas_efficiency:
            print(f"  cuBLAS efficiency: {cublas_efficiency:.1f}%")
    
    return results

def get_theoretical_peak_tflops():
    """Estimate theoretical peak TFLOPS for the current GPU (BF16 tensor cores)."""
    try:
        gpu_name = torch.cuda.get_device_name()
        # BF16 tensor core peak estimates
        peak_estimates = {
            'A100': 312.0,   # A100-80GB tensor TFLOPS (BF16)
            'H100': 989.0,   # H100 tensor TFLOPS (BF16)
            'V100': 0,       # V100 doesn't support BF16 tensor cores
            'RTX 4090': 165.0,  # RTX 4090 tensor TFLOPS (BF16)
            'RTX 3090': 0,      # RTX 3090 doesn't support BF16 tensor cores
            'RTX 4080': 113.0,  # RTX 4080 tensor TFLOPS (BF16) 
        }
        
        for gpu_type, peak in peak_estimates.items():
            if gpu_type in gpu_name:
                return peak if peak > 0 else None
        
        print(f"Unknown GPU: {gpu_name}, cannot estimate theoretical peak")
        return None
    except:
        return None

def profile_different_shapes():
    """Profile performance across different matrix sizes."""
    print("\n" + "="*80)
    print("PROFILING DIFFERENT SHAPES")
    print("="*80)
    
    shapes = [
        (1024, 1024),
        (2048, 2048), 
        (4096, 4096),
        (8192, 16384),
        (8192, 8192),
        (16384, 4096),  # Large size for stress testing
    ]
    
    results = []
    for shape in shapes:
        print(f"\nTesting shape: {shape}")
        try:
            result = benchmark_comparison(shape=shape, num_warmup=2, num_runs=2000, autotune_cublas=True)
            results.append((shape, result))
        except Exception as e:
            print(f"Failed to benchmark shape {shape}: {e}")
    
    # Summary table
    print(f"\n{'='*100}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*100}")
    print(f"{'Shape':<12} {'Custom':<18} {'PyTorch':<18} {'cuBLAS':<18} {'vs PyTorch':<12} {'vs cuBLAS':<10}")
    print(f"{'':12} {'(ms / TFLOPS)':<18} {'(ms / TFLOPS)':<18} {'(ms / TFLOPS)':<18} {'(speedup)':<12} {'(speedup)':<10}")
    print(f"{'-'*100}")
    
    for shape, result in results:
        shape_str = f"{shape[0]}x{shape[1]}"
        
        # Custom results
        custom_ms = result['custom_time_ms']
        custom_tflops = result['custom_tflops']
        custom_str = f"{custom_ms:.2f} / {custom_tflops:.1f}"
        
        # PyTorch results
        if result['torch_time_ms']:
            torch_ms = result['torch_time_ms']
            torch_tflops = result['torch_tflops']
            torch_str = f"{torch_ms:.2f} / {torch_tflops:.1f}"
            speedup_torch = result['speedup_vs_torch'] if result['speedup_vs_torch'] else 0
            speedup_torch_str = f"{speedup_torch:.2f}x"
        else:
            torch_str = "FAILED"
            speedup_torch_str = "N/A"
        
        # cuBLAS results
        if result['cublas_time_ms']:
            cublas_ms = result['cublas_time_ms']
            cublas_tflops = result['cublas_tflops']
            cublas_str = f"{cublas_ms:.2f} / {cublas_tflops:.1f}"
            speedup_cublas = result['speedup_vs_cublas'] if result['speedup_vs_cublas'] else 0
            speedup_cublas_str = f"{speedup_cublas:.2f}x"
        else:
            cublas_str = "FAILED"
            speedup_cublas_str = "N/A"
        
        print(f"{shape_str:<12} {custom_str:<18} {torch_str:<18} {cublas_str:<18} {speedup_torch_str:<12} {speedup_cublas_str:<10}")
    
    # Detailed PyTorch variant analysis for largest shape
    if results:
        print(f"\n{'='*80}")
        print("PYTORCH VARIANT ANALYSIS (for largest successful shape)")
        print(f"{'='*80}")
        
        # Find the largest shape that completed successfully
        largest_result = None
        largest_shape = None
        for shape, result in reversed(results):
            if result['torch_all_results']:
                largest_result = result
                largest_shape = shape
                break
        
        if largest_result and largest_shape:
            print(f"Shape: {largest_shape}")
            print(f"{'Variant':<15} {'Time (ms)':<12} {'TFLOPS':<10} {'vs Custom':<12}")
            print(f"{'-'*50}")
            
            custom_time_ms = largest_result['custom_time_ms']
            
            for variant, time_val in largest_result['torch_all_results'].items():
                if time_val is not None:
                    M, N, K = largest_shape[0], largest_shape[0], largest_shape[1]
                    flops_per_op = 2 * M * N * K
                    tflops = (flops_per_op / time_val) / 1e12
                    time_ms = time_val * 1000
                    speedup = time_val / (custom_time_ms / 1000)
                    
                    print(f"{variant:<15} {time_ms:<12.3f} {tflops:<10.2f} {speedup:<12.2f}x")
                else:
                    print(f"{variant:<15} {'FAILED':<12} {'N/A':<10} {'N/A':<12}")

def clear_algorithm_cache():
    """Clear the algorithm cache (useful for testing)."""
    global _algorithm_cache
    _algorithm_cache.clear()
    print("Algorithm cache cleared")

def benchmark_single_shape(shape=(4096, 4096), num_runs=100):
    """Convenience function to benchmark a single shape with detailed output."""
    print(f"Detailed benchmark for shape {shape}")
    return benchmark_comparison(shape=shape, num_warmup=5, num_runs=num_runs, autotune_cublas=True)

def benchmark_torch_compilation_variants(shape, num_runs=100, device='cuda', dtype=torch.bfloat16):
    """
    Benchmark PyTorch with different compilation modes if available.
    """
    print(f"Benchmarking PyTorch compilation variants...")
    
    results = {}
    
    # Standard PyTorch
    try:
        print(f"  Testing standard PyTorch...")
        time_standard = benchmark_torch_native_gemm(shape, num_runs, device, dtype, 'mm')
        results['standard'] = time_standard
        print(f"    Average time: {time_standard*1000:.4f} ms")
    except Exception as e:
        print(f"    Failed: {e}")
        results['standard'] = None
    
    # Try torch.compile if available (PyTorch 2.0+)
    try:
        print(f"  Testing torch.compile...")
        
        # Create a compiled function
        @torch.compile
        def compiled_mm(a, b):
            return torch.mm(a, b.t())
        
        # Warmup compilation
        A_warmup, B_warmup, _ = create_test_tensors(shape, device, dtype)
        _ = compiled_mm(A_warmup, B_warmup)
        torch.cuda.synchronize()
        
        # Benchmark compiled version
        times = []
        for _ in range(num_runs):
            A, B, _ = create_test_tensors(shape, device, dtype)
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            result = compiled_mm(A, B)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        time_compiled = np.mean(times)
        results['compiled'] = time_compiled
        print(f"    Average time: {time_compiled*1000:.4f} ms")
        
    except Exception as e:
        print(f"    torch.compile not available or failed: {e}")
        results['compiled'] = None
    
    return results

if __name__ == "__main__":
    # Check GPU capability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        print(f"GPU: {gpu_name}")
        major, minor = torch.cuda.get_device_capability()
        print(f"Compute Capability: {major}.{minor}")
        if major < 8:
            print("Warning: BF16 tensor cores require compute capability 8.0+")
        
        # Check PyTorch version for torch.compile availability
        torch_version = torch.__version__
        print(f"PyTorch Version: {torch_version}")
    
    # Profile different shapes
    profile_different_shapes()