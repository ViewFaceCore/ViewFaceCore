# ViewFaceCore 的 Nuget 包清单

## 核心包

- 核心包通过 `P/Invoke` 实现对了 [SeetaFace](https://github.com/SeetaFace6Open/index) 的封装, 此包只提供了通过图像的 `byte[]` 数据进行调用, 如果要更快捷方便的集成, 通常还需要安装针对你所使用的特定图形库的**扩展包** 

| **Nuget 包**                                                | **版本**                                                                                                | **描述**            |
|-------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|---------------------|
| [ViewFaceCore](https://www.nuget.org/packages/ViewFaceCore) | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.svg?color=%233F48CC&label=%20&style=flat-square) | ViewFaceCore 核心包 |


## 扩展包

- 扩展包依赖于核心包, 通过对核心包的扩展提供了不同的能力
- `ViewFaceCore.Extension.SystemDrawing` 图像扩展包的支持与 [System.Drawing.Common](https://www.nuget.org/packages/System.Drawing.Common/) 保持一致, 详细信息参见 [仅在 Windows 上支持 System.Drawing.Common](https://learn.microsoft.com/zh-cn/dotnet/core/compatibility/core-libraries/6.0/system-drawing-common-windows-only)

> 图形扩展包是针对特定图形库的 *ViewFaceCore* 的实现, 通常只需要安装所需的图像扩展包即可;  
> 若以下没有你使用的图形库扩展包, 可以 自行实现 ([图形扩展包开发指南](\ExtensionPackage_Guide.md)), [Issues](https://github.com/ViewFaceCore/ViewFaceCore/issues), [PR](https://github.com/ViewFaceCore/ViewFaceCore/pulls).  

| **Nuget 包**                                                                                                | **版本**                                                                                                                        | **描述**                  |
|-------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------------------------|
| [ViewFaceCore.Extension.ImageSharp](https://www.nuget.org/packages/ViewFaceCore.Extension.ImageSharp)       | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.Extension.ImageSharp.svg?color=%233F48CC&label=%20&style=flat-square)    | ImageSharp 图形扩展包     |
| [ViewFaceCore.Extension.SkiaSharp](https://www.nuget.org/packages/ViewFaceCore.Extension.SkiaSharp)         | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.Extension.SkiaSharp.svg?color=%233F48CC&label=%20&style=flat-square)     | SkiaSharp 图形扩展包      |
| [ViewFaceCore.Extension.SystemDrawing](https://www.nuget.org/packages/ViewFaceCore.Extension.SystemDrawing) | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.Extension.SystemDrawing.svg?color=%233F48CC&label=%20&style=flat-square) | System.Drawing 图形扩展包 |

> 当你在使用 *Asp.NET Core* 时, 可以通过此扩展包快速引入 `ViewFaceCore`.  

| **Nuget 包**                                                                                                            | **版本**                                                                                                                              | **描述**       |
|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|----------------|
| [ViewFaceCore.Extension.DependencyInjection](https://www.nuget.org/packages/ViewFaceCore.Extension.DependencyInjection) | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.Extension.DependencyInjection.svg?color=%233F48CC&label=%20&style=flat-square) | 依赖注入扩展包 |


## 运行时包

- 运行时包是 *SeetaFace* 类库, *P/Invoke* 的中间类库以及相关依赖的组合打包, 运行时包仅包含 *Native* 类库
- 在项目生成时运行时包会把 *Native* 类库输出到项目生成目录下的 `viewfacecore\{Platform}\{Architecture}` 文件夹下

| **Nuget 包**                                                                                                  | **版本**                                                                                                                         | **描述**                                                                                  |
|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| [ViewFaceCore.runtime.linux.arm64](https://www.nuget.org/packages/ViewFaceCore.runtime.linux.arm64)           | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.runtime.linux.arm64.svg?color=%233F48CC&label=%20&style=flat-square)      | Linux-arm64 运行时包, 支持树莓派, nanopi 等                                               |
| [ViewFaceCore.runtime.linux.arm](https://www.nuget.org/packages/ViewFaceCore.runtime.linux.arm)               | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.runtime.linux.arm.svg?color=%233F48CC&label=%20&style=flat-square)        | Linux-armhf 运行时包, 支持树莓派, nanopi 等                                               |
| [ViewFaceCore.runtime.ubuntu.20.04.x64](https://www.nuget.org/packages/ViewFaceCore.runtime.ubuntu.20.04.x64) | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.runtime.ubuntu.20.04.x64.svg?color=%233F48CC&label=%20&style=flat-square) | Linux-x64 运行时包, 支持 Ubuntu20.04+, Debian10+, Deepin20+ 等较新的 debian 系 Linux 系统 |
| [ViewFaceCore.runtime.win.x64](https://www.nuget.org/packages/ViewFaceCore.runtime.win.x64)                   | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.runtime.win.x64.svg?color=%233F48CC&label=%20&style=flat-square)          | Windows-x64 运行时包                                                                      |
| [ViewFaceCore.runtime.win.x86](https://www.nuget.org/packages/ViewFaceCore.runtime.win.x86)                   | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.runtime.win.x86.svg?color=%233F48CC&label=%20&style=flat-square)          | Windows-x86 运行时包                                                                      |

> 在没有安装 `VC++ Runtime` 环境的电脑运行时, 可以安装此包  

| **Nuget 包**                                                                              | **版本**                                                                                                                | **描述**              |
|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|-----------------------|
| [ViewFaceCore.runtime.win.vc](https://www.nuget.org/packages/ViewFaceCore.runtime.win.vc) | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.runtime.win.x86.svg?color=%233F48CC&label=%20&style=flat-square) | Windows VC++ 运行时包 |


## 模型包

- [SeetaFace 模型文件相关](https://github.com/SeetaFace6Open/index#%E7%99%BE%E5%BA%A6%E7%BD%91%E7%9B%98)
- 模型包是对 SeetaFace 的模型文件进行的 Nuget 打包, 模型包仅包含 *SeetaFace* 模型文件, 
- 在项目生成时模型包会把模型文件输出到项目生成目录下的 `viewfacecore\models` 文件夹下

| **Nuget 包**                                                                                                                | **版本**                                                                                                                                | **描述**                    |
|-----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|-----------------------------|
| [ViewFaceCore.model.age_predictor](https://www.nuget.org/packages/ViewFaceCore.model.age_predictor)                         | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.age_predictor.svg?color=%233F48CC&label=%20&style=flat-square)             | 年龄预测                    |
| [ViewFaceCore.model.eye_state](https://www.nuget.org/packages/ViewFaceCore.model.eye_state)                                 | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.eye_state.svg?color=%233F48CC&label=%20&style=flat-square)                 | 眼睛状态检测                |
| [ViewFaceCore.model.face_detector](https://www.nuget.org/packages/ViewFaceCore.model.face_detector)                         | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.face_detector.svg?color=%233F48CC&label=%20&style=flat-square)             | 人脸检测                    |
| [ViewFaceCore.model.face_landmarker_mask_pts5](https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_mask_pts5) | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.face_landmarker_mask_pts5.svg?color=%233F48CC&label=%20&style=flat-square) | 戴口罩关键定定位，5个关键点 |
| [ViewFaceCore.model.face_landmarker_pts5](https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_pts5)           | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.face_landmarker_pts5.svg?color=%233F48CC&label=%20&style=flat-square)      | 关键定定位，5个关键点       |
| [ViewFaceCore.model.face_landmarker_pts68](https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_pts68)         | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.face_landmarker_pts68.svg?color=%233F48CC&label=%20&style=flat-square)     | 关键定定位，68个关键点      |
| [ViewFaceCore.model.face_recognizer](https://www.nuget.org/packages/ViewFaceCore.model.face_recognizer)                     | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.face_recognizer.svg?color=%233F48CC&label=%20&style=flat-square)           | 人脸识别，68个关键点        |
| [ViewFaceCore.model.face_recognizer_light](https://www.nuget.org/packages/ViewFaceCore.model.face_recognizer_light)         | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.face_recognizer_light.svg?color=%233F48CC&label=%20&style=flat-square)     | 人脸识别，5个关键点         |
| [ViewFaceCore.model.face_recognizer_mask](https://www.nuget.org/packages/ViewFaceCore.model.face_recognizer_mask)           | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.face_recognizer_mask.svg?color=%233F48CC&label=%20&style=flat-square)      | 人脸识别，戴口罩            |
| [ViewFaceCore.model.fas_first](https://www.nuget.org/packages/ViewFaceCore.model.fas_first)                                 | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.fas_first.svg?color=%233F48CC&label=%20&style=flat-square)                 | 活体检测，局部              |
| [ViewFaceCore.model.fas_second](https://www.nuget.org/packages/ViewFaceCore.model.fas_second)                               | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.fas_second.svg?color=%233F48CC&label=%20&style=flat-square)                | 活体检测，全局              |
| [ViewFaceCore.model.gender_predictor](https://www.nuget.org/packages/ViewFaceCore.model.gender_predictor)                   | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.gender_predictor.svg?color=%233F48CC&label=%20&style=flat-square)          | 性别预测                    |
| [ViewFaceCore.model.mask_detector](https://www.nuget.org/packages/ViewFaceCore.model.mask_detector)                         | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.mask_detector.svg?color=%233F48CC&label=%20&style=flat-square)             | 口罩检测                    |
| [ViewFaceCore.model.pose_estimation](https://www.nuget.org/packages/ViewFaceCore.model.pose_estimation)                     | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.pose_estimation.svg?color=%233F48CC&label=%20&style=flat-square)           | 姿态检测                    |
| [ViewFaceCore.model.quality_lbn](https://www.nuget.org/packages/ViewFaceCore.model.quality_lbn)                             | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.quality_lbn.svg?color=%233F48CC&label=%20&style=flat-square)               | 质量检测                    |

> `ViewFaceCore.all_models` 包没有任何内容, 仅记录了依赖项, 安装此包时, 会通过依赖安装其它所有的模型包  

| **Nuget 包**                                                                      | **版本**                                                                                                           | **描述**   |
|-----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|------------|
| [ViewFaceCore.all_models](https://www.nuget.org/packages/ViewFaceCore.all_models) | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.all_models.svg?color=%233F48CC&label=%20&style=flat-square) | 所有模型包 |
