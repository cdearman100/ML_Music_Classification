#!/bin/bash

# Update package lists
apt-get update

# Install LLVM (required for llvmlite and numba)
apt-get install -y llvm llvm-dev clang

# Verify installation
llvm-config --version || echo "LLVM installation failed"

# Continue with normal deployment
pip install --upgrade pip
pip install -r requirements.txt
