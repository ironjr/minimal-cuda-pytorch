#include <torch/extension.h>

// Forward declaration of our CUDA function
torch::Tensor cuda_add(torch::Tensor a, torch::Tensor b);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Minimal CUDA extension example";
    m.def("cuda_add", &cuda_add, "Element-wise addition using CUDA");
} 