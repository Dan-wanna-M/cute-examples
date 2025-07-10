from gemm_kernels import debug_mbarrier

def main():
    print("Running debug mbarrier test...")
    debug_mbarrier.debug_mbar()
    print("Debug mbarrier test completed successfully.")

if __name__ == "__main__":
    main()
    