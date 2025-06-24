# Minimal PyTorch CUDA Extension Example

This is a minimal implementation of a PyTorch model that uses a single CUDA function written in C++ CUDA.

## Files

- `cuda_kernels.cu` - CUDA kernel implementation (element-wise addition)
- `bindings.cpp` - Python bindings using pybind11
- `setup.py` - Build configuration
- `model.py` - PyTorch model that uses the custom CUDA function
- `test.py` - Test script to verify the CUDA function works correctly

## Requirements

- PyTorch with CUDA support
- CUDA toolkit
- C++ compiler (gcc/g++ on Linux, MSVC on Windows)

## Build Instructions

1. Navigate to this directory:
   ```bash
   cd minimal_cuda_example
   ```

2. Build and install the extension:
   ```bash
   python setup.py install
   ```

   Or for development (builds in-place):
   ```bash
   python setup.py build_ext --inplace
   ```

## Usage

### Test the CUDA function
```bash
python test.py
```

### Run the model example
```bash
python model.py
```

## How it works

1. **CUDA Kernel** (`cuda_kernels.cu`): Implements a simple element-wise addition kernel that runs on GPU
2. **Python Bindings** (`bindings.cpp`): Uses pybind11 to expose the CUDA function to Python
3. **PyTorch Integration** (`model.py`): Wraps the CUDA function in a `torch.autograd.Function` for seamless integration with PyTorch's automatic differentiation
4. **Model Usage**: The custom CUDA function is used within a standard PyTorch model

## Extending this example

You can extend this example by:
- Adding more complex CUDA kernels
- Implementing custom backward passes
- Adding support for different data types
- Optimizing memory access patterns
- Adding error checking and validation

## Notes

- This example uses float32 tensors only
- The CUDA kernel is optimized for simplicity, not performance
- Error checking is minimal - production code should have more robust error handling 
