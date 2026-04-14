#!/bin/bash

set -e

echo "------------------------------------------------"
echo "Starting dependency installation..."
echo "------------------------------------------------"

# 1. Update system and install basic dependencies
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    liblapack-dev \
    libblas-dev \
    libarpack2-dev \
    libsuperlu-dev \
    libgflags-dev \
    libgoogle-perftools-dev \
    libprotobuf-dev \
    protobuf-compiler

# Navigate to tmp for compilation
cd /tmp

# 2. Install Armadillo
echo ">>> Installing Armadillo..."
if [ -d "armadillo-code" ]; then rm -rf armadillo-code; fi
git clone https://gitlab.com/conradsnicta/armadillo-code.git
cd armadillo-code
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo make install
cd /tmp && rm -rf armadillo-code

# 3. Install GoogleTest
echo ">>> Installing GoogleTest..."
if [ -d "googletest" ]; then rm -rf googletest; fi
git clone https://github.com/google/googletest.git
cd googletest
mkdir -p build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install
cd /tmp && rm -rf googletest

# 4. Install Google Logging
echo ">>> Installing glog..."
if [ -d "glog" ]; then rm -rf glog; fi
git clone https://github.com/google/glog.git
cd glog
mkdir -p build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install
cd /tmp && rm -rf glog

# 5. Install SentencePiece
echo ">>> Installing SentencePiece..."
if [ -d "sentencepiece" ]; then rm -rf sentencepiece; fi
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir -p build && cd build
cmake .. \
    -DSPM_ENABLE_SHARED=ON \
    -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
cd /tmp && rm -rf sentencepiece

# 6. Update dynamic linker run-time bindings
echo ">>> Running ldconfig..."
sudo ldconfig

echo "------------------------------------------------"
echo "Installation completed successfully!"
echo "------------------------------------------------"
