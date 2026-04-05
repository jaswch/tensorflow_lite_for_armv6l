<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_horizontal.png">
</div>

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4724125.svg)](https://doi.org/10.5281/zenodo.4724125)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1486/badge)](https://bestpractices.coreinfrastructure.org/projects/1486)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/tensorflow/tensorflow/badge)](https://securityscorecards.dev/viewer/?uri=github.com/tensorflow/tensorflow)
[![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow)
[![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow-py.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:tensorflow-py)
[![OSSRank](https://shields.io/endpoint?url=https://ossrank.com/shield/44)](https://ossrank.com/p/44)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)

**`Documentation`** |
------------------- |
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |

# TensorFlow Lite for armv6l
This Github fork provides instructions and pre-built binaries for ```tflite_runtime``` for armv6l SBCs.

## Supported Devices
1. Raspberry Pi 1 (Model B, Model A, Model B+, Model A+)
2. Raspberry Pi Zero
3. Raspberry Pi Zero (W, WH)

## Compiling
To compile this from scratch you must need to follow the prerequisites :
1. At least 40 GB of memory (RAM + swap)
2. Docker

Note :- This was done on a PC running Ubuntu 25.10 (Questing Quokka) 64-bit.

1. Clone the source ```https://github.com/jaswch/tensorflow.git```
2. It is recomended to use V2.16.1 ```git checkout v2.16.1```
3. Enter the project directory ```cd tensorflow/```
4. Now replace the contents of the docker file in ```/tensorflow/tensorflow/lite/tools/pip_package/Dockerfile.py3``` with the below code.
```python
ARG IMAGE
FROM ${IMAGE}
ARG PYTHON_VERSION
ARG NUMPY_VERSION

COPY update_sources.sh /
RUN /update_sources.sh

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
      build-essential \
      tzdata \
      software-properties-common \
      zlib1g-dev  \
      curl \
      wget \
      unzip \
      git && \
    apt-get clean

# Install Bazel.
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.15.0/bazelisk-linux-amd64 \
  -O /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel

# Install Python packages.
RUN dpkg --add-architecture armhf
RUN dpkg --add-architecture arm64
RUN yes | add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    apt-get install -y \
      python$PYTHON_VERSION \
      python$PYTHON_VERSION-dev \
      python$PYTHON_VERSION-venv \
      python$PYTHON_VERSION-distutils \
      libpython$PYTHON_VERSION-dev \
      libpython$PYTHON_VERSION-dev:armhf \
      libpython$PYTHON_VERSION-dev:arm64
RUN ln -sf /usr/bin/python$PYTHON_VERSION /usr/bin/python3
RUN curl -OL https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py
RUN rm get-pip.py
RUN pip3 install --upgrade pip
RUN pip3 install numpy~=$NUMPY_VERSION setuptools pybind11
RUN ln -sf /usr/include/python$PYTHON_VERSION /usr/include/python3
RUN ln -sf /usr/local/lib/python$PYTHON_VERSION/dist-packages/numpy/core/include/numpy /usr/include/python3/numpy
RUN curl -OL https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh cmake-3.28.1-linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

ENV CI_BUILD_PYTHON=python$PYTHON_VERSION
ENV CROSSTOOL_PYTHON_INCLUDE_PATH=/usr/include/python$PYTHON_VERSION

COPY with_the_same_user /
```
5. Now run this command to spin up a docker container and let it automatically build the python wheel package for you
```bash
sudo make -C tensorflow/lite/tools/pip_package docker-build   BASE_IMAGE=ubuntu:jammy   TENSORFLOW_TARGET=rpi0   PYTHON_VERSION=<PYTHON_VER>   EXTRA_CMAKE_FLAGS="-DCMAKE_BUILD_PARALLEL_LEVEL=2"
```
This will start an Ubuntu 22.04 (Jammy Jellyfish) container and build ```tflite_runtime``` for armv6l, you'll also have to replace ```<PYTHON_VER>``` by the version of python you want.

## Installation
Install th python wheel package ```.whl``` by using the command
```
pip3 install tflite_runtime-2.16.1-cp311-cp311m-linux_armv6l.whl
```

## Pre-built binaries
You can also get the pre-built python wheel packages by going to the releases tab


