#!/usr/bin/env bash

export BUILD_DIR=build.linux.x64
export BUILD_TYPE=Release
export PLATFORM_TARGET=x64

export ORZ_HOME=$(cd "$(dirname "$0")"; pwd)/../../build
export INSTALL_DIR=$(cd "$(dirname "$0")"; pwd)/../../build

HOME=$(cd `dirname $0`; pwd)

cd $HOME

mkdir "$BUILD_DIR"

cd "$BUILD_DIR"

mkdir "$INSTALL_DIR"

cmake "$HOME/.." \
-DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
-DPLATFORM="$PLATFORM_TARGET" \
-DORZ_ROOT_DIR="$ORZ_HOME" \
-DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

make -j16
make install

