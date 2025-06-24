.PHONY: build test clean install all

all: build test

build:
	@echo "Building CUDA extension..."
	python setup.py build_ext --inplace

install:
	@echo "Installing CUDA extension..."
	python setup.py install

test: build
	@echo "Running tests..."
	python test.py

model: build
	@echo "Running model example..."
	python model.py

clean:
	@echo "Cleaning build files..."
	rm -rf build/
	rm -rf *.egg-info/
	rm -f *.so
	rm -rf __pycache__/

help:
	@echo "Available targets:"
	@echo "  build   - Build the CUDA extension"
	@echo "  install - Install the CUDA extension"
	@echo "  test    - Run tests"
	@echo "  model   - Run model example"
	@echo "  clean   - Clean build files"
	@echo "  all     - Build and test (default)" 