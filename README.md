<div align="center">

# ViewFaceCore 
[![Nuget](https://img.shields.io/nuget/v/ViewFaceCore)](https://www.nuget.org/packages/ViewFaceCore/) &nbsp;&nbsp;
[![GitHub license](https://img.shields.io/github/license/ViewFaceCore/ViewFaceCore)](https://github.com/ViewFaceCore/ViewFaceCore/blob/main/LICENSE) &nbsp;&nbsp;
![GitHub stars](https://img.shields.io/github/stars/ViewFaceCore/ViewFaceCore?style=flat) &nbsp;&nbsp;
![GitHub forks](https://img.shields.io/github/forks/ViewFaceCore/ViewFaceCore)

<br/>

—— [💎 关于](#1-关于) &nbsp;| [⭐ 快速开始](#2-快速开始) &nbsp;| [🔧 二次开发](#3-二次开发) &nbsp;| [📄 基本说明](#4-基本说明) &nbsp;| [🐟 API](#5-viewfacecore-api) &nbsp;| [🔎 参考文档](#6-参考文档) &nbsp;| [❓ 常见问题](#7-常见问题) &nbsp;| [📦 使用许可](#8-使用许可) ——

</div>

## 1. 关于
- 一个基于 [SeetaFace6](https://github.com/SeetaFace6Open/index) 的 .NET 人脸识别解决方案
- 本项目受到了 [SeetaFaceEngine.Net](https://github.com/iarray/SeetaFaceEngine.Net) 的启发
- 开源、免费、跨平台 (win/linux)

## 2. 快速开始
### 2.1 受支持的 .NET 框架 和 操作系统  

   | 目标框架 |最低版本 | 操作系统 |
   | :-: |:-: | :-: |
   | .NET Framework |4.0 | win ( x64/x86 ) |
   | .NET Standard |2.0 | win ( x64/x86 ) |
   | .NET / .NET Core |3.1、5.0、6.0、7.0 | win ( x64/x86 )、linux ( arm/arm64/x64 ) |

### 2.2 简单的人脸信息检测  
以 Windows x64平台 为例，一个简单的人脸检测Demo。
1. 使用 [nuget](https://www.nuget.org) 安装依赖  

| 包名称 | 最小版本 | 生成文件夹 | 说明 |
| :- | :-: | - | - |
| [ViewFaceCore](https://www.nuget.org/packages/ViewFaceCore/) | [![](https://img.shields.io/nuget/v/ViewFaceCore.svg)](https://www.nuget.org/packages/ViewFaceCore) | —— | ViewFaceCore .NET 核心库 |
| [ViewFaceCore.all_models](https://www.nuget.org/packages/ViewFaceCore.all_models) | [![](https://img.shields.io/nuget/v/ViewFaceCore.all_models.svg)](https://www.nuget.org/packages/ViewFaceCore.all_models) | `viewfacecore\models` | 人脸检测的模型支持(图省事可以直接安装这个) |
| [ViewFaceCore.runtime.win.x64](https://www.nuget.org/packages/ViewFaceCore.runtime.win.x64) | [![](https://img.shields.io/nuget/v/ViewFaceCore.runtime.win.x64.svg)](https://www.nuget.org/packages/ViewFaceCore.runtime.win.x64) | `viewfacecore\win\x64` | Windows-x64 的本机运行时，其它平台自行选择安装，可安装多个 |
| [ViewFaceCore.Extension.SkiaSharp](https://www.nuget.org/packages/ViewFaceCore.Extension.SkiaSharp) | <span style="display:inline-block;width:150px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.Extension.SkiaSharp.svg)](https://www.nuget.org/packages/ViewFaceCore.Extension.SkiaSharp) </span> |  —— | SkiaSharp图像处理扩展，ImageSharp、SkiaSharp、System.Drawing三选一 |

2. 获取人脸信息  
```csharp
using SkiaSharp;
using System;
using ViewFaceCore.Core;
using ViewFaceCore.Model;

namespace ViewFaceCore.Demo.ConsoleApp
{
    internal class Program
    {
        private readonly static string imagePath = @"images/Jay_3.jpg";

        static void Main(string[] args)
        {
            using var bitmap = SKBitmap.Decode(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            FaceInfo[] infos = faceDetector.Detect(bitmap);
            Console.WriteLine($"识别到的人脸数量：{infos.Length} 个人脸信息：\n");
            Console.WriteLine($"No.\t人脸置信度\t位置信息");
            for (int i = 0; i < infos.Length; i++)
            {
                Console.WriteLine($"{i}\t{infos[i].Score:f8}\t{infos[i].Location}");
            }
            Console.ReadKey();
        }
    }
}
```

更多案例可以下载源码查看Demo。  

## 3. 二次开发
### 3.1 项目结构

```shell
├─Bridges                                        #Bridges
│  ├─Linux                                       ##Linux平台ViewFaceBridge项目
│  ├─Shared                                      ##共享库
│  └─Windows                                     ##Linux平台ViewFaceBridge项目
├─Examples                                       #一些案例
│  ├─ViewFaceCore.Demo.ConsoleApp                ##控制台项目案例
│  ├─ViewFaceCore.Demo.VideoForm                 ##Winform摄像头人脸识别项目
│  └─ViewFaceCore.Demo.WebApp                    ##ASP.NET Core web项目
├─Extensions                                     #扩展包项目
│  ├─ViewFaceCore.Extension.DependencyInjection  ##依赖注入扩展
│  ├─ViewFaceCore.Extension.ImageSharp           ##ImageSharp图像处理扩展项目
│  ├─ViewFaceCore.Extension.Shared               ##共享项目
│  ├─ViewFaceCore.Extension.SkiaSharp            ##SkiaSharp图像处理扩展项目
│  └─ViewFaceCore.Extension.SystemDrawing        ##System.Drawing图像处理扩展项目
├─Models                                         #模型项目
├─Runtimes                                       #对应各个平台的运行时
├─SeetaFace
│  └─index                                       #SeetaFace源码，build文件夹需要放到这个目录下面
├─Tests                                          #测试项目，包含各种单元测试
└─ViewFaceCore                                   #ViewFaceCore源码
```

### 3.2 开发环境
   - Visual Studio 2022，需要安装.NET4/4.5支持（[如何在Visual Studio 2022中安装.NET4/4.5？](https://www.quarkbook.com/?p=1311 "如何在Visual Studio 2022中安装.NET4/4.5？")）
   - Windows 10/11
   - Ubuntu 20.04 (WSL)、Debian 10/11等
   
### 3.3 编译SeetaFace6
#### 3.3.1 我对编译SeetaFace6不感兴趣~~~
即中科视图开源的SeetaFace6人脸识别引擎，如果你对编译这块不感兴趣，可以直接下载下面编译好的链接库，放入src/SeetaFace/index目录下面。  
二进制文件地址：https://github.com/ViewFaceCore/SeetaFace6OpenBinary/releases  
放置好之后的目录结构应该是这样的：  
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

####  3.3.2 我要编译SeetaFace6  
1. SeetaFace6官方编译方式  
参考：https://github.com/SeetaFace6Open/index#%E7%BC%96%E8%AF%91%E4%BE%9D%E8%B5%96
2. 使用项目中的编译脚本  
参考：https://github.com/ViewFaceCore/ViewFaceCore/blob/main/docs/SeetaFace_Build.md
	
### 3.4 编译SeetaFaceBridge
#### 3.4.1 编译Windows环境下的SeetaFaceBridge
通过3.3.1或者3.3.2小节，完成SeetaFace6编译之后，就可以在VS中直接右键->构建SeetaFaceBridge了。

#### 3.4.2 编译Linux环境下的SeetaFaceBridge
前提条件也是需要通过3.3.1或者3.3.2小节，完成SeetaFace6的编译。  
有三种方式可以编译Linux环境下的SeetaFaceBridge，WSL、Remote Linux、交叉编译。具体步奏就不一一介绍了，选择自己比较熟悉或者喜欢的方式，building~

### 3.5 编译ViewFaceCore
就像编译C#的Hello World一样简单。

## 4. 基本说明

### 4.1 各平台Runtime
为了兼容各个平台，我们分别编号好了各个平台的SeetaFace6以及ViewFaceBridge（用于通过C#调用C++写的SeetaFace6）。并打包好，发布到nuget上面。使用时，按需安装对应平台的nuget包。比如windows x64平台，安装对应的nuget包`ViewFaceCore.runtime.win.x64`。runtime可以同时安装多个平台，但是不能不安装，至少需要安装一个当前平台对应的runtime。  
支持的平台以及nuget包如下表所示：  

| Nuget包名  | 版本  |  输出目录  |  说明 |  
| ------------ | ------------ |------------ | ------------ |
| ViewFaceCore.runtime.win.x64  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.runtime.win.x64.svg)](https://www.nuget.org/packages/ViewFaceCore.runtime.win.x64) </span>  | `viewfacecore/win/x64`  |  Windows，x64  |
| ViewFaceCore.runtime.win.x86  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.runtime.win.x86.svg)](https://www.nuget.org/packages/ViewFaceCore.runtime.win.x86) </span>  | `viewfacecore/win/x86`|  Windows，x86  |
| ViewFaceCore.runtime.ubuntu.20.04.x64  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.runtime.ubuntu.20.04.x64.svg)](https://www.nuget.org/packages/ViewFaceCore.runtime.ubuntu.20.04.x64) </span>  | `viewfacecore/linux/x64`|  Linux，x64；支持Ubuntu20.04+、Debian10+、Deepin20+等较新的debian系Linux系统  |
| ViewFaceCore.runtime.linux.arm  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.runtime.linux.arm.svg)](https://www.nuget.org/packages/ViewFaceCore.runtime.linux.arm) </span>  | `viewfacecore/linux/arm`|  Linux，armhf；支持树莓派，nanopi等  |
| ViewFaceCore.runtime.linux.arm64  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.runtime.linux.arm64.svg)](https://www.nuget.org/packages/ViewFaceCore.runtime.linux.arm64) </span>  | `viewfacecore/linux/arm64`|  Linux，arm64；支持树莓派，nanopi等  |

在使用过程中，如果出现`DirectoryNotFoundException: Can not found library path.`之类的异常，请检查对应目录下是否存在runtime依赖，有时网络问题会导致nuget包下载失败。  


### 4.2 图形库扩展  
ViewFaceCore本身不依赖任何图形库，但是为了调用方便，我们开发了三个常见图形库扩展。如下表所示：  

| Nuget包名  | 版本  |  说明 |  
| ------------ | ------------ | ------------ |
| ViewFaceCore.Extension.SkiaSharp  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.Extension.SkiaSharp.svg)](https://www.nuget.org/packages/ViewFaceCore.Extension.SkiaSharp) </span>  |  SkiaSharp图形扩展  |
| ViewFaceCore.Extension.ImageSharp  |  <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.Extension.ImageSharp.svg)](https://www.nuget.org/packages/ViewFaceCore.Extension.ImageSharp) </span>  | ImageSharp图形扩展  |
| ViewFaceCore.Extension.SystemDrawing  |  <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.Extension.SystemDrawing.svg)](https://www.nuget.org/packages/ViewFaceCore.Extension.SystemDrawing) </span>  | System.Drawing图形扩展，微软不再支持System.Drawing跨平台了，但是这个包目前还是跨平台支持的  |

在使用过程中，可以选择任意一个自己熟悉或者当前项目中已经用到的图形库扩展使用。但是，**图形库扩展不能安装多个，只能三者选其一**！！！  

### 4.3 模型设置
> 在算法开发包中，除了代码库本身以外，还有数据文件，我们通常称之为模型。  
> 来源：入门教程 1.3节 http://leanote.com/blog/post/5e7d6cecab64412ae60016ef#title-11  

我们封装好了每个人脸识别功能对应的模型，并打包发布至nuget。在使用时，只需要安装对应模型的nuget包即可。  
nuget包如下表所示:  

| Nuget包名  | 版本  |  说明 |  
| ------------ | ------------ | ------------ |
| ViewFaceCore.all_models  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.all_models.svg)](https://www.nuget.org/packages/ViewFaceCore.all_models) </span>  |  包含全部模型。不知道装哪一个时，装这个就行了，但是包特别大（300多MB）  |
| ViewFaceCore.model.age_predictor  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.model.age_predictor.svg)](https://www.nuget.org/packages/ViewFaceCore.model.age_predictor) </span>  |  年龄预测  |
| ViewFaceCore.model.eye_state  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.model.eye_state.svg)](https://www.nuget.org/packages/ViewFaceCore.model.eye_state) </span>  |  眼睛状态检测  |
| ViewFaceCore.model.gender_predictor  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.model.gender_predictor.svg)](https://www.nuget.org/packages/ViewFaceCore.model.gender_predictor) </span>  |  性别预测  |
| ViewFaceCore.model.face_detector  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.model.face_detector.svg)](https://www.nuget.org/packages/ViewFaceCore.model.face_detector) </span>  |  人脸检测  |
| ViewFaceCore.model.mask_detector  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.model.mask_detector.svg)](https://www.nuget.org/packages/ViewFaceCore.model.mask_detector) </span>  |  口罩检测  |
| ViewFaceCore.model.face_landmarker_mask_pts5  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.model.face_landmarker_mask_pts5.svg)](https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_mask_pts5) </span>  |  戴口罩关键定定位，5个关键点  |
| ViewFaceCore.model.face_landmarker_pts5  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.model.face_landmarker_pts5.svg)](https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_pts5) </span>  |  关键定定位，5个关键点  |
| ViewFaceCore.model.face_landmarker_pts68  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.model.face_landmarker_pts68.svg)](https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_pts68) </span>  |  关键定定位，68个关键点  |
| ViewFaceCore.model.face_recognizer  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.model.face_recognizer.svg)](https://www.nuget.org/packages/ViewFaceCore.model.face_recognizer) </span>  |  人脸识别，68个关键点  |
| ViewFaceCore.model.face_recognizer_light  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.model.face_recognizer_light.svg)](https://www.nuget.org/packages/ViewFaceCore.model.face_recognizer_light) </span>  |  人脸识别，5个关键点  |
| ViewFaceCore.model.face_recognizer_mask  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.model.face_recognizer_mask.svg)](https://www.nuget.org/packages/ViewFaceCore.model.face_recognizer_mask) </span>  |  人脸识别，戴口罩  |
| ViewFaceCore.model.fas_first  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.model.fas_first.svg)](https://www.nuget.org/packages/ViewFaceCore.model.fas_first) </span>  |  活体检测，局部  |
| ViewFaceCore.model.fas_second  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.model.fas_second.svg)](https://www.nuget.org/packages/ViewFaceCore.model.fas_second) </span>  |  活体检测，全局  |
| ViewFaceCore.model.pose_estimation  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.model.pose_estimation.svg)](https://www.nuget.org/packages/ViewFaceCore.model.pose_estimation) </span>  |  姿态检测  |
| ViewFaceCore.model.quality_lbn  | <span style="display:inline-block;width:100px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.model.quality_lbn.svg)](https://www.nuget.org/packages/ViewFaceCore.model.quality_lbn) </span>  |  质量检测  |

### 4.4 调用说明  
#### 4.4.1 对象生命周期  
这里的对象的生命周期指的是人脸识别中各个功能对象的生命周期，并不是C#中GC和对象的生命周期。虽然也和C#中对象生命周期密不可分，但是这并不是这一小节的主题，这里不会过多的解释C#语言本身的特性。  
用`FaceDetector`举个例子。在`FaceDetector`的构造函数中  
```csharp
public FaceDetector(FaceDetectConfig config = null)
{
    this.DetectConfig = config ?? new FaceDetectConfig();
    _handle = ViewFaceNative.GetFaceDetectorHandler(this.DetectConfig.FaceSize
        , this.DetectConfig.Threshold
        , this.DetectConfig.MaxWidth
        , this.DetectConfig.MaxHeight
        , (int)this.DetectConfig.DeviceType);
    if (_handle == IntPtr.Zero)
    {
        throw new Exception("Get face detector handler failed.");
    }
}
```
通过Native调用的方式，调用C++项目ViewFaceBridge中的函数`GetFaceDetectorHandler`获取SeetaFace6中`seeta::v6::FaceDetector`对象的IntPtr句柄。  
ViewFaceBridge中的函数`GetFaceDetectorHandler`函数代码如下：  
```cpp
View_Api seeta::v6::FaceDetector *GetFaceDetectorHandler(const double faceSize = 20, const double threshold = 0.9, const double maxWidth = 2000, const double maxHeight = 2000, const SeetaDevice deviceType = SEETA_DEVICE_AUTO)
{
	seeta::v6::FaceDetector *faceDetector = new seeta::v6::FaceDetector(ModelSetting(modelPath + "face_detector.csta", deviceType));
	faceDetector->set(FaceDetector::Property::PROPERTY_MIN_FACE_SIZE, faceSize);
	faceDetector->set(FaceDetector::Property::PROPERTY_THRESHOLD, threshold);
	faceDetector->set(FaceDetector::Property::PROPERTY_MAX_IMAGE_WIDTH, maxWidth);
	faceDetector->set(FaceDetector::Property::PROPERTY_MAX_IMAGE_HEIGHT, maxHeight);
	return faceDetector;
}
```
当对象使用完毕后，`FaceDetector`中Dispose方法中通过Native调用`DisposeFaceDetector`函数，释放掉`seeta::v6::FaceDetector`对象。  
```cpp
View_Api void DisposeFaceDetector(seeta::v6::FaceDetector *handler)
{
	_dispose(handler);
}
```
综上所述，在编写代码的过程中，**一定要使用`using`语句或在结束后调用`Dispose`释放掉对象**。且SeetaFace6对象的构造和释放会比较耗时，其中涉及到模型加载、计算等，建议**尽可能的复用对象以及在需要频繁使用新对象的场景中使用对象池。**  
对象复用，又涉及到线程安全的问题。更多关于线程安全的细节，请继续阅读下一节。  

#### 4.4.2 线程安全  
> 线程安全也是开发中需要重点关注的特性。然而，线程安全在不同的上下文解释中总会有不同解释。为了避免理解的偏差，这里用几种不同的用例去解释识别器的使用。  
>   
> 1. 对象可以跨线程传递。线程1构造的识别器，可以在线程2中调用。  
> 2. 对象的构造可以并发构造，即可以多个线程同时构造识别器。  
> 3. 单个对象的接口调用不可以并发调用，即单个对象，在多个线程同时使用是被禁止的。  
> 来源：入门教程 1.5 线程安全性 http://leanote.com/blog/post/5e7d6cecab64412ae60016ef#title-11    

因为SeetaFace6本身不支持多线程调用，所以在这个库设计的时候，在每个不支持并发操作的功能中通过加锁限制并发调用。可以认为，在单个对象的不同操作中，是线程安全的。  

#### 4.4.3 初始化配置
在一些场景下，比如不支持AVX2指令集、需要拿取内部日志等场景下，默认设置并不能满足要求。为此，我们提供了一个全局配置项：`GlobalConfig`，下面的小节将具体介绍支持的特性。

##### 4.4.3.1 输出内部日志
在生产环境或者某些不方便调试场景下，又出现一些莫名其妙的问题的时候，不妨看看内部日志，说不定有不一样的收获。
```csharp
static void Config()
{
    //打印内部日志
    GlobalConfig.SetLog((msg) =>
    {
        Console.WriteLine($"[内部日志]{msg}");
    });
}
```

##### 4.4.3.2 特定指令集支持
x86环境，默认情况下，使用支持AVX2、FMA指令集的tennis神经网络推理系统。但在一些低功耗CPU上面，比如Intel的J系列和N系列，阉割了AVX2指令集。在这些不支持AVX2或FMA指令集的CPU上面运行时，可能会报异常：0x00007FFC3FDD104E (tennis.dll) (ConsoleApp1.exe 中)处有未经处理的异常: 0xC000001D: IllegInstruction。  
原因是tennis使用了不支持的指令集。下表是tennis文件对应支持的指令集。  

| 文件  | 指令集  | 说明  |
| ------------ | ------------ | ------------ |
| tennis.dll  | AVX2、FMA  | 默认  |
| tennis_haswell.dll  | AVX2、FMA   |   |
| tennis_sandy_bridge.dll  | AVX2   |   |
| tennis_pentium.dll  | SSE2   |   |

但是tennis同样提供了不同指令集上面的解决方案。ViewFaceCore通过一个全局配置项，可以强制使用支持具体指令集的tennis。  
如下所示：  
```csharp
static void Config()
{
    //设置只支持SSE2指令集
    GlobalConfig.SetInstruction(X86Instruction.SSE2);
}
```
需要注意的是，设置指令集支持，必需在初始化任何API之前，否者无效。

## 5. ViewFaceCore API

### 5.1 所有API通用配置参数  
下表是所有API都能使用的配置参数，有些参数可能并不会生效。  

| 配置项  | 类型  |  默认值 | 说明  |
| ------------ |------------ | ------------ | ------------ |
| DeviceType  | 枚举；支持值：AUTO、CPU、GPU   | AUTO  | 检测所用的设备类型，目前只提供CPU支持，需要GPU请自行编译[TenniS](https://github.com/TenniS-Open/TenniS "TenniS")  |

### 5.2 FaceAntiSpoofing（活体检测）  
活体检测API。  
活体检测识别器可以加载一个`局部检测模型`或者`局部检测模型+全局检测模型`，使用参数`Global`来区分，默认为`True`。  
当使用`局部检测模型`时，需要安装模型`ViewFaceCore.model.fas_second`。 当使用`局部检测模型+全局检测模型`时，需要安装模型`ViewFaceCore.model.fas_first`和`ViewFaceCore.model.fas_second`。  

**配置项`FaceAntiSpoofingConfig`**  

| 配置项  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| VideoFrameCount  | int  | 10  | 设置视频帧数，用于`PredictVideo`，一般来说，在10以内，帧数越多，结果越稳定，相对性能越好，但是得到结果的延时越高。  |
| BoxThresh  | float  | 0.8  | 攻击介质存在的分数阈值，该阈值越高，表示对攻击介质的要求越严格  |
| Threshold  | FaceAntiSpoofingConfigThreshold  | (0.3, 0.8)  | 活体识别时，如果清晰度(clarity)低的话，就会直接返回FUZZY。清晰度满足阈值，则判断真实度（reality），超过阈值则认为是真人，低于阈值是攻击。  |
| Global  | bool  | true  | 是否开启全局检测模型。**在Linux平台下面，存在bug，无法设置为FALSE** |

FaceAntiSpoofingConfigThreshold：  

| 配置项  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| Clarity  | float  | 0.3  | 清晰度(clarity)，如果清晰度(clarity)低的话，就会直接返回FUZZY  |
| Reality  | float  | 0.8  | 真实度(clarity)，超过阈值则认为是真人，低于阈值是攻击。  |

**AntiSpoofing**  
单帧活体检测。  
```csharp
public AntiSpoofingResult Predict(FaceImage image, FaceInfo info, FaceMarkPoint[] points)
```
入参：  

| 名称  | 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| image  | FaceImage  | object  | - | 图像数据  |
| info  | FaceInfo  | object  | - | 人脸信息  |
| points  | FaceMarkPoint[]  | struct数组  | - | 关键点位  |

出参`AntiSpoofingResult`：  

| 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| AntiSpoofingStatus  | 枚举  | - | Error（错误或没有找到指定的人脸索引处的人脸）、Real（真实人脸）、Spoof（攻击人脸（假人脸））、Fuzzy（无法判断（人脸成像质量不好））、Detecting（正在检测）  
| Clarity  | float  | - | 清晰度  |
| Reality  | float  | - | 真实度  |

**调用示例**  
单帧活体检测。  
```csharp
static void AntiSpoofingDemo()
{
    using var bitmap = SKBitmap.Decode(imagePath0);

    using FaceDetector faceDetector = new FaceDetector();
    using FaceLandmarker faceMark = new FaceLandmarker();
    using FaceAntiSpoofing faceAntiSpoofing = new FaceAntiSpoofing();

    var info = faceDetector.Detect(bitmap).First();
    var markPoints = faceMark.Mark(bitmap, info);

    Stopwatch sw = Stopwatch.StartNew();
    sw.Start();

    var result = faceAntiSpoofing.Predict(bitmap, info, markPoints);
    Console.WriteLine($"活体检测，结果：{result.Status}，清晰度:{result.Clarity}，真实度：{result.Reality}，耗时：{sw.ElapsedMilliseconds}ms");

    sw.Stop();
    Console.WriteLine();
}
```

**AntiSpoofingVideo**  
视频帧识别。  
```csharp
public AntiSpoofingResult AntiSpoofingVideo(FaceImage image, FaceInfo info, FaceMarkPoint[] points)
```
使用方式同上。  

### 5.3 FaceDetector（人脸检测）
人脸检测，输入待检测的图片，输出检测到的每个人脸位置，用矩形表示。  
人脸检测需要模型`ViewFaceCore.model.face_detector`。一般检测返回的所有人脸的人脸位置数组，并按照置信度从大大小进行排序返回。  

**配置项`FaceDetectConfig`**  

| 配置项  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| FaceSize  | int  | 20  | 最小人脸，最小人脸和检测器性能息息相关。主要方面是速度，使用建议上，我们建议在应用范围内，这个值设定的越大越好。  |
| Threshold  | double  | 0.9  |检测器阈值。默认值是0.9，合理范围为[0, 1]。这个值一般不进行调整，除了用来处理一些极端情况。这个值设置的越小，漏检的概率越小，同时误检的概率会提高。  |
| MaxWidth  | int  | 2000  | 可检测的图像最大宽度  |
| MaxHeight  | int  | 2000  | 可检测的图像最大高度  |

更多细节请参考：http://leanote.com/blog/post/5e7d6cecab64412ae60016ef#title-14  

**Detect**  
人脸信息检测。  
```csharp
public FaceInfo[] Detect(FaceImage image)
```
入参：  

| 名称  | 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| image | FaceImage  | object  | -  | 人脸图像信息数据  |

出参：  

| 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| FaceInfo  | struct数组  | -  | 人脸信息  |

FaceInfo：  

| 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| Score  | float  | -  | 人脸置信度  |
| Location  | FaceRect  | -  | 人脸位置  |

FaceRect：  

| 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| X  | int  | -  | 左上角点横坐标  |
| Y  | int  | -  | 左上角点纵坐标  |
| Width  | int  | -  | 矩形宽度  |
| Height  | int  | -  | 矩形高度  |

**调用示例**  
识别人脸并标记出来。  
```csharp
using System;
using System.Drawing;
using System.Linq;
using ViewFaceCore;
using ViewFaceCore.Core;
using ViewFaceCore.Model;

namespace Demo
{
    internal class Program
    {
        private readonly static string imagePath = @"images/Jay_3.jpg";
        private readonly static string outputPath = @"images/Jay_out.jpg";

        static void Main(string[] args)
        {
            using var bitmap = (Bitmap)Image.FromFile(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            FaceInfo[] infos = faceDetector.Detect(bitmap);
            //输出人脸信息
            Console.WriteLine($"识别到的人脸数量：{infos.Length} 个人脸信息：\n");
            Console.WriteLine($"No.\t人脸置信度\t位置信息");
            for (int i = 0; i < infos.Length; i++)
            {
                Console.WriteLine($"{i}\t{infos[i].Score:f8}\t{infos[i].Location}");
            }
            //画方框，标记人脸
            using (Graphics g = Graphics.FromImage(bitmap))
            {
                g.DrawRectangles(new Pen(Color.Red, 4), infos.Select(p => new RectangleF(p.Location.X, p.Location.Y, p.Location.Width, p.Location.Height)).ToArray());
            }
            bitmap.Save(outputPath);
            Console.WriteLine($"输出图片已保存至：{outputPath}");
            Console.WriteLine();
        }
    }
}
```

### 5.4 FaceLandmarker（人脸关键点定位器）
关键定定位输入的是原始图片和人脸检测结果，给出指定人脸上的关键点的依次坐标。  
这里检测到的5点坐标循序依次为，左眼中心、右眼中心、鼻尖、左嘴角和右嘴角。注意这里的左右是基于图片内容的左右，并不是图片中人的左右，即左眼中心就是图片中左边的眼睛的中心。  

**配置项`FaceLandmarkConfig`**  

| 配置项  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| MarkType  | 枚举  | Light  | 关键点类型  |

MarkType枚举：  

| 枚举值  | 所选模型  | 说明  |
| ------------ | ------------ | ------------ | 
| Normal  | ViewFaceCore.model.face_landmarker_pts68  | 68个关键点检测模型  |
| Light  | ViewFaceCore.model.face_landmarker_pts5  | 5个关键点检测模型  |
| Mask  | ViewFaceCore.model.face_landmarker_mask_pts5  | 戴口罩关键点检测模型  |

需要注意的是：  

> 这里的关键点是指人脸上的关键位置的坐标，在一些表述中也将关键点称之为特征点，但是这个和人脸识别中提取的特征概念没有任何相关性。**并不存在结论，关键点定位越多，人脸识别精度越高。**  
> 一般的关键点定位和其他的基于人脸的分析是基于5点定位的。而且算法流程确定下来之后，只能使用5点定位。5点定位是后续算法的先验，并不能直接替换。**从经验上来说，5点定位已经足够处理人脸识别或其他相关分析的精度需求，单纯增加关键点个数，只是增加方法的复杂度，并不对最终结果产生直接影响。**  
> 来源：入门教程 2.2 人脸关键点定位器 http://leanote.com/blog/post/5e7d6cecab64412ae60016ef#title-15  

**Mark**  
```csharp
public FaceMarkPoint[] Mark(FaceImage image, FaceInfo info) 
```
入参：  

| 名称  | 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| image | FaceImage  | object  | -  | 人脸图像信息数据  |
| info | FaceInfo  | struct  | -  | 面部信息  |

出参：  

| 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| FaceMarkPoin[]  | struct数组  | -  | 关键点坐标，坐标循序依次为，左眼中心、右眼中心、鼻尖、左嘴角和右嘴角  |

FaceMarkPoin  

| 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| X  | int  | -  | 左上角点横坐标  |
| Y  | int  | -  | 左上角点纵坐标  |

**调用示例**  
识别人脸并标记出来。  
```csharp
static void FaceMarkDemo()
{
    using var bitmap0 = SKBitmap.Decode(imagePath0);
    using var faceImage = bitmap0.ToFaceImage();
    using FaceDetector faceDetector = new FaceDetector();
    using FaceLandmarker faceMark = new FaceLandmarker();
    Stopwatch sw = new Stopwatch();

    var infos = faceDetector.Detect(faceImage);
    var markPoints = faceMark.Mark(faceImage, infos[0]);

    sw.Stop();
    Console.WriteLine($"识别到的关键点个数：{markPoints.Length}，耗时：{sw.ElapsedMilliseconds}ms");
    foreach (var item in markPoints)
    {
        Console.WriteLine($"X:{item.X}, Y:{item.Y}");
    }
    Console.WriteLine();
}
```

### 5.5 FaceRecognizer（人脸特征提取和对比）
人脸识别的一个基本概念，就是将待识别的人脸经过处理变成二进制数据的特征，然后基于特征表示的人脸进行相似度计算，最终与相似度阈值对比，一般超过阈值就认为特征表示的人脸是同一个人。  

**配置项`FaceRecognizeConfig`**  

| 配置项  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| FaceType  | 枚举  | Normal  | 人脸识别模型  |
| Threshold  | 阈值  | FaceType.Normal：0.62、FaceType.Mask：0.4、FaceType.Light：0.55 | 不同人脸识别模型具有不同的默认值  |

配置项中Threshold为私有变量，需要通过方法`GetThreshold`来获取阈值，通过`SetThreshold`来设置阈值。  

**Extract**  
特征提取。  
```csharp
public float[] Extract(FaceImage image, FaceMarkPoint[] points)
```

入参：  

| 名称 | 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| image  | FaceImage  | object  | -  | 人脸图像信息数据  |
| points  | FaceMarkPoint数组  | -  | 人脸标记点位，通过FaceLandmarker.Mark方法获取  |

出参：  

| 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| float[]  | array  | -  | 特征值  |

提取的特征值都是float数组。提取特征值后通过下面的`Compare`方法和其他人脸特征值进行对比，特征对比方式是向量內积。  

![](https://docs.geeiot.net/server/index.php?s=/api/attachment/visitFile&sign=edd1b0d73da5c2daaa5e4fd609ed1b10)

**Compare**  
计算相似度。  
```csharp
public float Compare(float[] lfs, float[] rfs)
```

入参：  

| 名称 | 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| lfs | float[]  | array  | -  | 特征值  |
| rfs | float[]  | array  | -  | 特征值  |

出参：  

| 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| float  | float  | -  | 特征值相似度  |

**IsSelf**  
判断是否为同一人。  
```csharp
public bool IsSelf(float similarity)
或
public bool IsSelf(float[] lfs, float[] rfs)
```

入参：  

| 名称 | 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| lfs | float[]  | array  | -  | 特征值  |
| rfs | float[]  | array  | -  | 特征值  |

或  

| 名称 | 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| similarity | float  | float  | -  | 特征值相似度  |

通过和设置的阈值对比，大于阈值则为同一人。  

出参：  

| 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| bool  | bool  | -  | true为同一人，false不是同一个人  |

**调用示例**  
提取两张图片特征值后，判断两张图片中的人脸是否为同一人。  
```csharp
static void FaceRecognizerDemo()
{
    Stopwatch sw = Stopwatch.StartNew();
    sw.Start();

    using var faceImage0 = SKBitmap.Decode(imagePath0).ToFaceImage();
    using var faceImage1 = SKBitmap.Decode(imagePath1).ToFaceImage();
    //检测人脸信息
    using FaceDetector faceDetector = new FaceDetector();
    FaceInfo[] infos0 = faceDetector.Detect(faceImage0);
    FaceInfo[] infos1 = faceDetector.Detect(faceImage1);
    //标记人脸位置
    using FaceLandmarker faceMark = new FaceLandmarker();
    FaceMarkPoint[] points0 = faceMark.Mark(faceImage0, infos0[0]);
    FaceMarkPoint[] points1 = faceMark.Mark(faceImage1, infos1[0]);
    //提取特征值
    using FaceRecognizer faceRecognizer = new FaceRecognizer();
    float[] data0 = faceRecognizer.Extract(faceImage0, points0);
    float[] data1 = faceRecognizer.Extract(faceImage1, points1);
    //对比特征值
    bool isSelf = faceRecognizer.IsSelf(data0, data1);

    Console.WriteLine($"识别到的人脸是否为同一人：{isSelf}，对比耗时：{sw.ElapsedMilliseconds}ms");
    Console.WriteLine();
    sw.Stop();
}
```

### 5.6 FaceTracker（人脸追踪）
人脸追踪是在进行识别之前就利用视频特性，首先就确认在视频序列中出现的那些人是同一人，并获取人脸在视频中的位置。人脸追踪获取的结果（`FaceTrackInfo`）可以直接转换成`FaceInfo`使用。  
`FaceTrackInfo`相比于`FaceInfo`多了一个PID字段，PID就是人员编号，对于视频中出现的人脸，如果跟踪分配了同一个PID，那么就可以认为相同PID的人脸属于同一个人。  
更多内容请查看：http://leanote.com/blog/post/5e7d6cecab64412ae60016ef#title-29

**配置项`FaceTrackerConfig`**  

| 配置项  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| Width  | int  | -  | 视频宽度  |
| Height  | int  | - | 视频高度  |
| MinFaceSize  | int  | 20 | 设置可检测的人脸大小，为人脸宽和高乘积的二次根值。最小人脸和检测器性能息息相关。主要方面是速度，使用建议上，我们建议在应用范围内，这个值设定的越大越好。  |
| Threshold  | float  | 0.9 | 检测器阈值。合理范围为[0, 1]。这个值一般不进行调整，除了用来处理一些极端情况。这个值设置的越小，漏检的概率越小，同时误检的概率会提高。  |
| Stable  | bool  | false | 是否进行检测结果的帧间平滑，使得检测结果从视觉上更好一些。  |
| Interval  | int  | 10 | 检测间隔  |

配置项`FaceTrackerConfig`必须指定视频宽度和高度，不能为空。  

**Track**  
识别传入图像中的人脸，并返回可跟踪的人脸信息。  
```csharp
public FaceTrackInfo[] Track(FaceImage image)
```

入参：  

| 名称 | 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| image | FaceImage  | struct  | -  | 要追踪的图像  |

出参：  

| 参数  | 类型  |  默认值 | 说明  |
| ------------ | ------------ | ------------ | ------------ |
| FaceTrackInfo  | struct数组  | -  | 人脸信息  |

**Reset**  
当检测逻辑断开，或者切换视频的时候，就需要排除之前跟踪的逻辑，这个时候调用Reset方式清楚之前所有跟踪的结果，重新PID计数。  
```csharp
public void Reset()
```

调用示例：  
追踪一张图片中的人脸，并获取人脸标记点。  
```csharp
static void FaceTrackDemo()
{
    using var faceImage = SKBitmap.Decode(imagePath0).ToFaceImage();
    using FaceLandmarker faceMark = new FaceLandmarker();
    using FaceTracker faceTrack = new FaceTracker(new FaceTrackerConfig(faceImage.Width, faceImage.Height));
    var result = faceTrack.Track(faceImage);
    if (result == null || !result.Any())
    {
        Console.WriteLine("未追踪到任何人脸！");
        return;
    }
    foreach (var item in result)
    {
        FaceInfo faceInfo = item.ToFaceInfo();
        //标记人脸
        var points = faceMark.Mark(faceImage, faceInfo);
    }
}
```

### 5.7 MaskDetector（口罩检测）  

#### 5.7.1 口罩检测  
用于检测是否戴了口罩或有遮挡。  

#### 5.7.2 戴口罩人脸识别  
口罩人脸识别，其底层还是调用口罩人脸识别模块，只需要替换为口罩人脸识别模型。  

### 5.8 FaceQuality（质量检测）

### 5.9 AgePredictor（年龄预测）

### 5.10 GenderPredictor（性别预测）

### 5.11 EyeStateDetector（眼睛状态检测）


## 6. 参考文档
- [*SeetaFace6 说明*](https://github.com/seetafaceengine/SeetaFace6/blob/master/README.md)
- [*SeetaFace 各接口说明*](https://github.com/seetafaceengine/SeetaFace6/tree/master/docs)
- [*SeetaFace 入门教程*](http://leanote.com/blog/post/5e7d6cecab64412ae60016ef)


## 7. 常见问题
1. Unable to load DLL 'ViewFaceBridge' or one of its dependencies  
	1. 检查nuget包是否下载完全，编译目标文件夹下面的viewfacecore文件夹中是否有对应平台的依赖文件，比如说windows x64平台，在viewfacecore文件夹下面应该会有win/x64文件夹，文件夹中有很多*.dll文件。  
	2. 缺少vc++依赖，安装nuget包`ViewFaceCore.runtime.win.vc`.[![](https://img.shields.io/nuget/v/ViewFaceCore.runtime.win.vc.svg)](https://www.nuget.org/packages/ViewFaceCore.runtime.win.vc)  

2. 开始人脸识别时卡死，然后异常结束，或者报异常：0x00007FFC3FDD104E (tennis.dll) (ConsoleApp1.exe 中)处有未经处理的异常: 0xC000001D: IllegInstruction。  
	参考：https://github.com/ViewFaceCore/ViewFaceCore/tree/main#4432-%E7%89%B9%E5%AE%9A%E6%8C%87%E4%BB%A4%E9%9B%86%E6%94%AF%E6%8C%81

## 8. 使用许可   
<div align="center">

[Copyright (c) 2021, View](https://github.com/ViewFaceCore/ViewFaceCore/blob/main/LICENSE) | [*Copyright (c) 2019, SeetaTech*](https://github.com/SeetaFace6Open/index/blob/master/LICENSE)

</div>

> [\[源\]](https://github.com/SeetaFace6Open/index#%E8%81%94%E7%B3%BB%E6%88%91%E4%BB%AC) > *`SeetaFace` 开源版可以免费用于商业和个人用途。如果需要更多的商业支持，请联系商务邮件 bd@seetatech.com*
