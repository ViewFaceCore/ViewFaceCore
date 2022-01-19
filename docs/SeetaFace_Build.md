# SeetaFace6 Build

用于在各个平台编译[SeetaFace6](https://github.com/SeetaFace6Open/index "SeetaFace6")。以及收集并修复了官方仓库中的一些错误。

### 编译方法
#### Linux

Linux脚本来源：https://blog.ofortune.xyz/2020/08/30/seetaface6-intro/
1. 安装编译工具（Debian系）
```shell
sudo apt install git gcc g++ cmake -y
```

2. 下载源码  
```shell
git clone --recursive https://github.com/ViewFaceCore/ViewFaceCore.git
```

3. 授权编译脚本  
```shell
cd ViewFaceCore/docs/seeta-face6-build
sudo chmod +x build.*.sh
```

4. 开始编译（x64平台）
```shell
sudo ./build.linux.x64.sh
```

#### Winodws

1. 安装VS 2019编译工具  
需要选择MSVC v142 VS2019工作负载，如图所示：
![](https://raw.githubusercontent.com/ViewFaceCore/ViewFaceCore/dev/docs/seeta-face6-build/docs/vs.png)

2. 安装[jom](https://download.qt.io/official_releases/jom/ "jom")  
下载[jom](https://download.qt.io/official_releases/jom/ "jom")，解压后将jom.exe所在目录设置为系统环境变量。  

2. 双击对应架构脚本开始编译  

### 修复内容

#### 1. QualityOfLBN对象不管是动态创建还是new创建，传过去的ModelSetting参数都会出现异常，取不到model数组的值。
**来自：** https://github.com/seetafaceengine/SeetaFace6/issues/33  
**修复方式：**
把头文件`QualityOfLBN.h`的51行
```cpp
QualityOfLBN(const seeta::ModelSetting &setting = seeta::ModelSetting())
```
改为
```cpp
QualityOfLBN( const SeetaModelSetting &setting )
```

源文件`QualityOfLBN.cpp`的728行
```cpp
QualityOfLBN::QualityOfLBN(const seeta::ModelSetting &setting)
```
改为
```cpp
QualityOfLBN::QualityOfLBN( const SeetaModelSetting &setting )
```

#### 2. 活体检测错误  
**来自：** https://github.com/seetafaceengine/SeetaFace6/issues/22  
**修复方式：**  
修改`FaceAntiSpoofing.h`的38行  
```cpp
explicit FaceAntiSpoofing( const seeta::ModelSetting &setting );
```
改为  
```cpp
explicit FaceAntiSpoofing( const SeetaModelSetting &setting );
```

修改`FaceAntiSpoofing.cpp`的1235行  
```cpp
FaceAntiSpoofing::FaceAntiSpoofing( const seeta::ModelSetting &setting )
```
改为  
```cpp
FaceAntiSpoofing::FaceAntiSpoofing( const SeetaModelSetting &setting )
```

#### 3. win10,vs2019,vc14环境下编译OpenRoleZoo报错

**来自：** https://github.com/SeetaFace6Open/index/issues/4  
**修复方式：**  
修改代码`OpenRoleZoo/include/orz/mem/pot.h`，在第9行`#include<memory>`后面插入一行`#include <functional>`补充所需要的头文件。
