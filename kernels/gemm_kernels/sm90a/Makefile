# Makefile for CUDA debug_mbarrier project

# Compiler settings
NVCC = nvcc
CXX = g++
TARGET = debug_mbarrier_test
CUDA_FILE = debug_mbarrier.cu
MAIN_FILE = debug.cpp
HEADER_FILE = debug_mbarrier.cuh

# CUDA and system paths (adjust as needed)
CUDA_PATH =/usr/local/cuda
CUTLASS_PATH =../../../libs/cutlass

# Compiler flags
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr
CXX_FLAGS = -O3 -std=c++17

# Include paths
INCLUDES = -I$(CUDA_PATH)/include -I$(CUTLASS_PATH)/include -I.

# Library paths and libraries
LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcuda

# Source files
SOURCES = $(CUDA_FILE) $(MAIN_FILE)
OBJECTS = debug_mbarrier.o main.o

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) -o $(TARGET) $(LIBS)

# Compile CUDA source file
debug_mbarrier.o: $(CUDA_FILE) $(HEADER_FILE)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $(CUDA_FILE) -o debug_mbarrier.o

# Compile main C++ file
main.o: $(MAIN_FILE)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $(MAIN_FILE) -o main.o

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Debug build
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: CXX_FLAGS += -g -DDEBUG
debug: $(TARGET)

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || echo "NVCC not found in PATH"
	@nvidia-smi || echo "nvidia-smi not found"

# Install dependencies (example for Ubuntu/Debian)
install-deps:
	@echo "Installing CUDA development dependencies..."
	sudo apt-get update
	sudo apt-get install nvidia-cuda-toolkit nvidia-cuda-dev

# Help target
help:
	@echo "Available targets:"
	@echo "  all       - Build the executable (default)"
	@echo "  clean     - Remove build artifacts"
	@echo "  run       - Build and run the program"
	@echo "  debug     - Build with debug symbols"
	@echo "  check-cuda - Check CUDA installation"
	@echo "  install-deps - Install CUDA dependencies (Ubuntu/Debian)"
	@echo "  help      - Show this help message"

.PHONY: all clean run debug check-cuda install-deps help