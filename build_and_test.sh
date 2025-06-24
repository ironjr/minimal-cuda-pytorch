#!/bin/bash

echo "Building minimal CUDA extension example..."

# Clean previous builds
rm -rf build/
rm -rf minimal_cuda_example.egg-info/
rm -f *.so

# Build the extension
echo "Building extension..."
python setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    
    echo "Running tests..."
    python test.py
    
    if [ $? -eq 0 ]; then
        echo "✓ Tests passed!"
        
        echo "Running model example..."
        python model.py
        
        if [ $? -eq 0 ]; then
            echo "✓ Model example completed successfully!"
        else
            echo "✗ Model example failed!"
        fi
    else
        echo "✗ Tests failed!"
    fi
else
    echo "✗ Build failed!"
fi 