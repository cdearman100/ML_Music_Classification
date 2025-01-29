#!/bin/bash

# Update package lists
apt-get update

# Install LLVM
apt-get install -y llvm-dev

# Continue with normal deployment
pip install -r requirements.txt
