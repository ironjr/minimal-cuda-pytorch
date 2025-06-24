#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for element-wise addition
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host function that launches the CUDA kernel
torch::Tensor cuda_add(torch::Tensor a, torch::Tensor b) {
    // Check that inputs are on CUDA
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
    
    // Create output tensor
    auto c = torch::zeros_like(a);
    
    // Get tensor info
    int n = a.numel();
    float* a_ptr = a.data_ptr<float>();
    float* b_ptr = b.data_ptr<float>();
    float* c_ptr = c.data_ptr<float>();
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    add_kernel<<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, n);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    return c;
} 