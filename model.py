import torch
import torch.nn as nn
try:
    import minimal_cuda_example
except ImportError:
    print("Please build the CUDA extension first by running:")
    print("python setup.py install")
    exit(1)

class CudaAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        return minimal_cuda_example.cuda_add(a, b)
    
    @staticmethod
    def backward(ctx, grad_output):
        # For element-wise addition, gradient passes through unchanged
        return grad_output, grad_output

class MinimalCudaModel(nn.Module):
    def __init__(self, input_size):
        super(MinimalCudaModel, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        
    def forward(self, x):
        # Apply first linear layer
        out1 = self.linear1(x)
        
        # Apply second linear layer  
        out2 = self.linear2(x)
        
        # Use our custom CUDA function to add them
        result = CudaAddFunction.apply(out1, out2)
        
        return result

def test_model():
    # Create model and move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MinimalCudaModel(128).to(device)
    
    # Create test input
    x = torch.randn(32, 128, device=device)
    
    # Forward pass
    output = model(x)
    
    # Simple loss for testing
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    print(f"Model output shape: {output.shape}")
    print(f"Loss: {loss.item()}")
    print("Custom CUDA function successfully integrated!")

if __name__ == "__main__":
    test_model() 