#!/usr/bin/env bash

# From https://blog.ofortune.xyz/2020/08/30/seetaface6-intro/

# 环境变量准备
env_setup() {
  BUILD_HOME=$(
    cd "$(dirname "$0")" || exit
    pwd
  )

  BUILD_PATH_BASE="$BUILD_HOME"/build
  BUILD_PATH_SEETA="$BUILD_PATH_BASE"/temp
  INSTALL_PATH_SEETA="$BUILD_PATH_BASE"/SeetaFace
  SOURCE_PATH_SEETA="$BUILD_HOME"/../../src/SeetaFace/index

  mkdir -p "$BUILD_PATH_BASE"
  mkdir -p "$BUILD_PATH_SEETA"
  mkdir -p "$INSTALL_PATH_SEETA"

  export BUILD_HOME
  export BUILD_PATH_BASE
  export BUILD_PATH_SEETA
  export INSTALL_PATH_SEETA
  export SOURCE_PATH_SEETA

  CORES="-j"$(grep -c "processor" </proc/cpuinfo)
  export CORES
  
  echo "BUILD_HOME: $BUILD_HOME"
  echo "Build Path: $BUILD_PATH_BASE"
  
  if [ -d $BUILD_PATH_BASE ]; then 
    echo 'Delete existing build target folder...';  
    rm -rf $BUILD_PATH_BASE;
  fi
}

build_seeta_OpenRoleZoo() {
  echo -e "\n>> Building OpenRoleZoo"
  mkdir -p "$BUILD_PATH_SEETA"/OpenRoleZoo
  cd "$BUILD_PATH_SEETA"/OpenRoleZoo || exit

  echo -e ">>> Fixing OpenRoleZoo"
  if grep -q "<functional>" "$SOURCE_PATH_SEETA"/OpenRoleZoo/include/orz/mem/pot.h; then
    echo -e ">>>> Already Fixed!"
  else
    sed -i '/<memory>/a\#include <functional>' "$SOURCE_PATH_SEETA"/OpenRoleZoo/include/orz/mem/pot.h
    echo ">>>> Fixed!"
  fi

  echo -e ">>> Configuring OpenRoleZoo"
  cmake "$SOURCE_PATH_SEETA"/OpenRoleZoo \
    -DCMAKE_BUILD_TYPE="Release" \
    -DORZ_WITH_OPENSSL=OFF \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH_SEETA" || exit

  echo -e "\n>>> Making OpenRoleZoo"
  make "$CORES" || exit
  make install || exit

  echo -e "\n>> OpenRoleZoo Built"
}

build_seeta_Authorize() {
  echo -e "\n>> Building SeetaAuthorize"
  mkdir -p "$BUILD_PATH_SEETA"/SeetaAuthorize
  cd "$BUILD_PATH_SEETA"/SeetaAuthorize || exit

  echo -e ">>> Configuring SeetaAuthorize"
  cmake "$SOURCE_PATH_SEETA"/SeetaAuthorize \
    -DCMAKE_BUILD_TYPE="Release" \
    -DPLATFORM="auto" \
    -DORZ_ROOT_DIR="$INSTALL_PATH_SEETA" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH_SEETA" || exit

  echo -e "\n>>> Making SeetaAuthorize"
  make "$CORES" || exit
  make install || exit

  echo -e "\n>> SeetaAuthorize Built"
}

build_seeta_TenniS() {
  echo -e "\n>> Building TenniS"
  mkdir -p "$BUILD_PATH_SEETA"/TenniS
  cd "$BUILD_PATH_SEETA"/TenniS || exit

  echo -e ">>> Configuring TenniS"
  
  # 这里我选用了Arm平台，其它平台根据文档修改
  cmake "$SOURCE_PATH_SEETA"/TenniS \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH_SEETA" \
    -DTS_USE_FAST_MATH=ON \
    -DTS_USE_NEON=ON \
    -DTS_BUILD_TEST=OFF \
    -DTS_BUILD_TOOLS=OFF \
    -DTS_ON_ARM=ON || exit

  echo -e "\n>>> Making TenniS"
  make "$CORES" || exit
  make install || exit

  echo -e "\n>> TenniS Built"
}

build_seeta_module() {
  echo -e "\n>> Building $1"
  mkdir -p "$BUILD_PATH_SEETA"/"$1"
  cd "$BUILD_PATH_SEETA"/"$1" || exit

  echo -e ">>> Configuring $1"
  cmake "$SOURCE_PATH_SEETA"/"$1" \
    -DCMAKE_BUILD_TYPE="Release" \
    -DPLATFORM="auto" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH_SEETA" \
    -DSEETA_INSTALL_PATH="$INSTALL_PATH_SEETA" \
    -DORZ_ROOT_DIR="$INSTALL_PATH_SEETA" \
    -DCMAKE_MODULE_PATH="$INSTALL_PATH_SEETA"/cmake \
    -DCONFIGURATION="Release" \
    -DSEETA_AUTHORIZE=OFF \
    -DSEETA_MODEL_ENCRYPT=ON || exit

  echo -e "\n>>> Making $1"
  make "$CORES" || exit
  make install || exit

  echo -e "\n>> $1 Built"
}

build_seeta() {
  echo -e "\n> Building SeetaFace"

  build_seeta_OpenRoleZoo
  build_seeta_Authorize
  build_seeta_TenniS
  
  # 需要注意的是如果是64位平台，部分.so文件会放置于/lib64目录下，而其它模块的cmake配置写的有问题，会搜索不到，需要手动复制到/lib目录下。（同时编译64位和32位时需要注意这个操作是有问题的）
  if [ -d "$INSTALL_PATH_SEETA"/lib64 ]; then
    cp -u "$INSTALL_PATH_SEETA"/lib64/* "$INSTALL_PATH_SEETA"/lib
  fi
  
  # 这里只编译了4个模块，按需增加
  build_seeta_module Landmarker
  build_seeta_module FaceRecognizer6
  build_seeta_module PoseEstimator6
  build_seeta_module QualityAssessor3

  build_seeta_module FaceAntiSpoofingX6
  build_seeta_module FaceBoxes
  build_seeta_module FaceTracker6
  build_seeta_module SeetaAgePredictor
  build_seeta_module SeetaEyeStateDetector
  build_seeta_module SeetaGenderPredictor
  build_seeta_module SeetaMaskDetector

  # 前面说过了，QualityAssessor3等模块会忽略我们手动设定的INSTALL_PREFIX，我们手动移动合并一下文件
  cp -rfu "$SOURCE_PATH_SEETA"/build/* "$INSTALL_PATH_SEETA"
  rm -rf "$SOURCE_PATH_SEETA"/build
  
  # 一样的问题
  if [ -d "$INSTALL_PATH_SEETA"/lib64 ]; then
    cp -u "$INSTALL_PATH_SEETA"/lib64/* "$INSTALL_PATH_SEETA"/lib
  fi

  if [ -d "$INSTALL_PATH_SEETA"/lib/auto ]; then
    cp -u "$INSTALL_PATH_SEETA"/lib/auto/* /"$INSTALL_PATH_SEETA"/lib
  fi
}

echo "==========Start Building=========="

env_setup
build_seeta

echo "==========Build Finished!=========="
