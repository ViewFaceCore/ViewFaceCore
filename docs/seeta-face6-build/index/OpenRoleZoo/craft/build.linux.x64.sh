#!/usr/bin/env bash

export BUILD_DIR=build.linux.x64
export BUILD_TYPE=Release
export PLATFORM_TARGET=x64

export WITH_SSL=OFF

export INSTALL_DIR=$(cd "$(dirname "$0")"; pwd)/../../build

HOME=$(cd `dirname $0`; pwd)

echo $HOME

cd $HOME

mkdir "$BUILD_DIR"

cd "$BUILD_DIR"

cmake "$HOME/.." \
-DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
-DPLATFORM="$PLATFORM_TARGET" \
-DORZ_WITH_OPENSSL="$WITH_SSL" \
-DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

make -j16
make install


