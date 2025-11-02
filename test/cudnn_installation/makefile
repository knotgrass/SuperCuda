# Makefile for checking cuDNN-frontend installation

# Compiler
NVCC = nvcc

# Paths (adjust these according to your conda environment)
CONDA_ENV = $(CONDA_PREFIX)
CUDA_PATH = $(CONDA_ENV)
CUDNN_PATH = $(CONDA_ENV)

# Compiler flags
NVCC_FLAGS = -std=c++17
PYTHON_INCLUDE = $(shell python -c "import site; print(site.getsitepackages()[0])")/include
INCLUDE_FLAGS = -I$(CUDNN_PATH)/include -I$(CUDA_PATH)/include -I$(PYTHON_INCLUDE)
LIBRARY_FLAGS = -L$(CUDNN_PATH)/lib -L$(CUDA_PATH)/lib64 -lcudnn

# Target executable
TARGET = check_cudnn_frontend

# Source file
SOURCE = check_cudnn_frontend.cu

# Build rule
all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) $(LIBRARY_FLAGS) $(SOURCE) -o $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build files
clean:
	rm -f $(TARGET)

.PHONY: all run clean
