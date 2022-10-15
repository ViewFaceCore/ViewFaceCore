# SeetaFace6 构建

## SeetaFace6官方编译方式

  > 参考：https://github.com/SeetaFace6Open/index#%E7%BC%96%E8%AF%91%E4%BE%9D%E8%B5%96


## 使用本项目编译的类库

  - 下载编译好的[动态库](https://github.com/ViewFaceCore/SeetaFace6OpenBinary/releases)，放入 `src/SeetaFace/index` 目录下面。  
  - 目录结构示意

    ```shell
    src
    └─SeetaFace
       └─index
          └─build
             ├─bin
             │  ├─x64
             │  └─x86
             ├─cmake
             ├─include
             ├─lib
             │  ├─arm
             │  ├─arm64
             │  ├─x64
             │  └─x86
             └─lib64
    ```


## 使用项目中的编译脚本

  用于在各个平台编译 [SeetaFace6](https://github.com/SeetaFace6Open/index "SeetaFace6"), 以及收集并修复了官方仓库中的一些错误

### Linux
  
  > Linux 脚本来源：https://blog.ofortune.xyz/2020/08/30/seetaface6-intro/

  1. 安装编译工具（Debian系）
  
  ```shell
  sudo apt install git gcc g++ cmake -y
  ```
  > PS: 如果 armhf 无法安装 g++, 提示 `libtirpc-dev : Depends: libtirpc3 (= 1.3.1-1) but 1.3.1-1+deb11u1 is to be installed`, 下载 [armhf](https://packages.debian.org/bullseye/armhf/libtirpc3/download), 使用 `dkpg -i` 重新安装  
  
  2. 下载ViewFaceCore源码  

  ```shell
  git clone https://github.com/ViewFaceCore/ViewFaceCore.git
  ```
  
  3. 下载SeetaFace6源码  

  ```shell
  mkdir ViewFaceCore/src/SeetaFace && cd ViewFaceCore/src/SeetaFace
  git clone --recursive https://github.com/ViewFaceCore/index.git
  ```
  
  4. 授权编译脚本  

  ```shell
  cd ../../scripts/SeetaFace6/
  sudo chmod +x build.*.sh
  ```
  
  5. 开始编译（x64平台）

  ```shell
  sudo ./build.linux.x64.sh
  ```
  
### Winodws
  
  1. 安装 `Visual Studio 2019` 编译工具
  
     需要选择MSVC v142 VS2019工作负载, 如图所示：
  <!-- ![](/assets/logos/vs.png) -->
        
  <image src="../assets/logos/vs.png" height=400></image>
  
  2. 下载 `ViewFaceCore` 源码  

     ```shell
     git clone https://github.com/ViewFaceCore/ViewFaceCore.git
     ```
  
  3. 下载 `SeetaFace6` 源码
  
     > 在 `ViewFaceCore` 目录下启动终端

     ```shell
     mkdir src/SeetaFace                                               #创建 SeetaFace 根目录
     cd src/SeetaFace                                                  #进入 SeetaFace 根目录
     git clone --recursive https://github.com/ViewFaceCore/index.git   #下载源码
     ```
  
  4. 双击对应架构脚本开始编译  

     > 在项目ViewFaceCore根目录下面的`scripts/SeetaFace6`文件中找到`build.win.vc.x64.bat`或`build.win.vc.x86.bat`双击打开或CMD中打开, 开始编译
  
### 修复内容
  
  1. `QualityOfLBN` 对象不管是动态创建还是 `new` 创建, 传过去的 `ModelSetting` 参数都会出现异常, 取不到 `model` 数组的值

     - 来自 https://github.com/seetafaceengine/SeetaFace6/issues/33  
     - 方案
     
       修改 `QualityOfLBN.h` 的 51 行
        ```cpp
        - QualityOfLBN(const seeta::ModelSetting &setting = seeta::ModelSetting())
        + QualityOfLBN(const SeetaModelSetting &setting)
        ```
    
       修改 `QualityOfLBN.cpp` 的 728 行
        ```cpp
        - QualityOfLBN::QualityOfLBN(const seeta::ModelSetting &setting)
        + QualityOfLBN::QualityOfLBN(const SeetaModelSetting &setting)
        ```
  
  2. 活体检测错误

     - 来自 https://github.com/seetafaceengine/SeetaFace6/issues/22  
     - 方案  
       修改 `FaceAntiSpoofing.h` 的 38 行  
       ```cpp
       - explicit FaceAntiSpoofing(const seeta::ModelSetting &setting);
       + explicit FaceAntiSpoofing(const SeetaModelSetting &setting);
       ```
       
       修改 `FaceAntiSpoofing.cpp` 的 1235 行  
       ```cpp
       - FaceAntiSpoofing::FaceAntiSpoofing(const seeta::ModelSetting &setting)
       + FaceAntiSpoofing::FaceAntiSpoofing(const SeetaModelSetting &setting)
       ```
  
  3. 眼睛状态检测错误

     - 方案

       修改 `EyeStateDetector.h` 的 16 行  
       ```cpp
       - SEETA_API explicit EyeStateDetector(const seeta::ModelSetting &setting);
       + SEETA_API explicit EyeStateDetector(const SeetaModelSetting &setting);
       ```
       
       修改 `EyeStateDetector.cpp` 的 653 行  
       ```cpp
       - EyeStateDetector::EyeStateDetector(const seeta::ModelSetting &setting)
       + EyeStateDetector::EyeStateDetector(const SeetaModelSetting &setting)
       ```
  
  4. 口罩识别错误

     - 方案
  
       修改 `MaskDetector.h` 的 17 行  
       ```cpp
       - SEETA_API explicit MaskDetector(const seeta::ModelSetting &setting = seeta::ModelSetting());
       + SEETA_API explicit MaskDetector(const SeetaModelSetting &setting);
       ```
       
       修改 `MaskDetector.cpp` 的 427 行  
       ```cpp
       - MaskDetector::MaskDetector(const seeta::ModelSetting &setting)
       + MaskDetector::MaskDetector(const SeetaModelSetting &setting)
       ```
  
  5. win10,vs2019,vc14环境下编译OpenRoleZoo报错
  
     - 来自 https://github.com/SeetaFace6Open/index/issues/4  
     - 方案
  
       修改 `OpenRoleZoo/include/orz/mem/pot.h`
       
       > 在 `#include<memory>` 后面插入一行 `#include <functional>` 补充所需要的头文件

       ```cpp
         #include<memory>
       + #include <functional>
       ```
       
  6. TenniS Windows下编译报错

       - 报错 `'towlower': is not a member of 'std'`  
       - 方案

         修改 `TenniS\src\compiler\argparse.cpp` 的 21 行
         ```cpp
         - std::transform(cvt.begin(), cvt.end(), cvt.begin(), std::towlower);
         + std::transform(cvt.begin(), cvt.end(), cvt.begin(), std::tolower);
         ```

         添加头文件引用
         ```cpp
         + #include <algorithm>
         ```
  