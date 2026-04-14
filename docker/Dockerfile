FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    # Armadillo dependencies
    liblapack-dev \
    libblas-dev \
    libarpack2-dev \
    libsuperlu-dev \
    # Google Logging & SentencePiece dependencies
    libgflags-dev \
    libgoogle-perftools-dev \
    libprotobuf-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

# Install Armadillo
RUN git clone https://gitlab.com/conradsnicta/armadillo-code.git \
    && cd armadillo-code \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && cd /tmp && rm -rf armadillo-code

# Install GoogleTest
RUN git clone https://github.com/google/googletest.git \
    && cd googletest \
    && mkdir build && cd build \
    && cmake .. -DBUILD_SHARED_LIBS=ON \
    && make -j$(nproc) \
    && make install \
    && cd /tmp && rm -rf googletest

# Install Google Logging
RUN git clone https://github.com/google/glog.git \
    && cd glog \
    && mkdir build && cd build \
    && cmake .. -DBUILD_SHARED_LIBS=ON \
    && make -j$(nproc) \
    && make install \
    && cd /tmp && rm -rf glog

# Install SentencePiece
RUN git clone https://github.com/google/sentencepiece.git \
    && cd sentencepiece \
    && mkdir build && cd build \
    && cmake .. \
        -DSPM_ENABLE_SHARED=ON \
        -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && make install \
    && cd /tmp && rm -rf sentencepiece

RUN ldconfig

WORKDIR /project

CMD ["/bin/bash"]
