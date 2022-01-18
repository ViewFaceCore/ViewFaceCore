#!/usr/bin/env bash

export BUILD_DIR=build.linux.x64_gpu
export BUILD_TYPE=Release
export PLATFORM_TARGET=x64

export PLATFORM=x64
export INSTALL_DIR=$(cd "$(dirname "$0")"; pwd)/../../build

HOME=$(cd `dirname $0`; pwd)

cd $HOME

mkdir "$BUILD_DIR"

cd "$BUILD_DIR"


cmake "$HOME/.." \
-DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
-DCONFIGURATION="$BUILD_TYPE" \
-DPLATFORM="$PLATFORM_TARGET" \
-DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
-DTS_USE_CUDA=ON \
-DTS_USE_CUBLAS=ON \
-DTS_USE_OPENMP=ON \
-DTS_USE_SIMD=ON \
-DTS_ON_HASWELL=ON

make -j16

make install
