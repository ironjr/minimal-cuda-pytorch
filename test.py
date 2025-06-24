import torch
try:
    import minimal_cuda_example
except ImportError:
    print("Please build the CUDA extension first by running:")
    print("python setup.py install")
    exit(1)

def test_cuda_add():
    print("Testing CUDA addition function...")
    
    # Create test tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = torch.randn(1000, device=device)
    b = torch.randn(1000, device=device)
    
    # Test our CUDA function
    c_cuda = minimal_cuda_example.cuda_add(a, b)
    
    # Compare with PyTorch's built-in addition
    c_torch = a + b
    
    # Check if results are close
    max_diff = torch.max(torch.abs(c_cuda - c_torch)).item()
    print(f"Maximum difference between CUDA and PyTorch: {max_diff}")
    
    if max_diff < 1e-6:
        print("✓ CUDA function produces correct results!")
    else:
        print("✗ CUDA function has incorrect results!")
    
    # Test with different shapes
    shapes = [(10,), (10, 10), (5, 20, 30)]
    for shape in shapes:
        a = torch.randn(shape, device=device)
        b = torch.randn(shape, device=device)
        c_cuda = minimal_cuda_example.cuda_add(a, b)
        c_torch = a + b
        max_diff = torch.max(torch.abs(c_cuda - c_torch)).item()
        print(f"Shape {shape}: max difference = {max_diff}")

if __name__ == "__main__":
    test_cuda_add() 