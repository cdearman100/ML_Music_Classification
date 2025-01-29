#!/bin/bash

# Update package lists
apt-get update

# Install LLVM (required for llvmlite and numba)
apt-get install -y llvm llvm-dev clang

# Install llvmlite from a pre-built wheel
pip install --find-links https://anaconda.org/numba/llvmlite/files llvmlite

# Continue with normal deployment
pip install -r requirements.txt
