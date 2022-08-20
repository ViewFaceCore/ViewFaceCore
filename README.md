<div align="center">

# ViewFaceCore 
[![Nuget](https://img.shields.io/nuget/v/ViewFaceCore)](https://www.nuget.org/packages/ViewFaceCore/) &nbsp;&nbsp;
[![GitHub license](https://img.shields.io/github/license/ViewFaceCore/ViewFaceCore)](https://github.com/ViewFaceCore/ViewFaceCore/blob/main/LICENSE) &nbsp;&nbsp;
![GitHub stars](https://img.shields.io/github/stars/ViewFaceCore/ViewFaceCore?style=flat) &nbsp;&nbsp;
![GitHub forks](https://img.shields.io/github/forks/ViewFaceCore/ViewFaceCore)

<br/>

—— [📄 关于](#📄&nbsp;关于) &nbsp;| [⭐ 快速开始](#⭐&nbsp;快速开始) &nbsp;| [🔧 构建](#🔧&nbsp;构建) &nbsp;| [📄 文档](#📄&nbsp;文档) &nbsp;| [❓ 常见问题](#❓&nbsp;常见问题) &nbsp;| [📦 使用许可](#📦&nbsp;使用许可) ——

</div>

## 📄&nbsp;关于
- 一个基于 [SeetaFace6](https://github.com/SeetaFace6Open/index) 的 .NET 人脸识别解决方案
- 本项目受到了 [SeetaFaceEngine.Net](https://github.com/iarray/SeetaFaceEngine.Net) 的启发
- 开源、免费、跨平台 (win/linux)

## ⭐ 快速开始
- ### 受支持的 .NET 框架 和 操作系统  


   | 目标框架 |最低版本 | 操作系统 |
   | :-: |:-: | :-: |
   | .NET Framework |4.0 | win ( x64/x86 ) |
   | .NET Standard |2.0 | win ( x64/x86 ) |
   | .NET / .NET Core |3.1、5.0、6.0、7.0 | win ( x64/x86 )、linux ( arm/arm64/x64 ) |

- ### 简单的人脸信息检测  
   - 以 Windows x64 为例  
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

- ### 扩展包说明    
	- #### 人脸识别模型  
	
	- #### 图形扩展  
	
	| Nuget包名  | 版本  |  说明 |  
	| ------------ | ------------ | ------------ |
	| ViewFaceCore.Extension.SkiaSharp  | <span style="display:inline-block;width:150px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.Extension.SkiaSharp.svg)](https://www.nuget.org/packages/ViewFaceCore.Extension.SkiaSharp) </span>  |  SkiaSharp图形扩展  |
	| ViewFaceCore.Extension.ImageSharp  |  <span style="display:inline-block;width:150px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.Extension.ImageSharp.svg)](https://www.nuget.org/packages/ViewFaceCore.Extension.ImageSharp) </span>  | ImageSharp图形扩展  |
	| ViewFaceCore.Extension.SystemDrawing  |  <span style="display:inline-block;width:150px"> [![](https://img.shields.io/nuget/v/ViewFaceCore.Extension.SystemDrawing.svg)](https://www.nuget.org/packages/ViewFaceCore.Extension.SystemDrawing) </span>  | System.Drawing图形扩展，微软不再支持System.Drawing跨平台了，但是这个包目前还是跨平台支持的  |
	

## 🔧 开发
- ### **项目结构**
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

- ### **开发环境**
   - Visual Studio 2022，需要安装.NET4/4.5支持（[如何在Visual Studio 2022中安装.NET4/4.5？](https://www.quarkbook.com/?p=1311 "如何在Visual Studio 2022中安装.NET4/4.5？")）
   - Windows 10/11
   - Ubuntu 20.04 (WSL)、Debian 10/11等
   
- ### **编译SeetaFace6**
	- #### 我对编译SeetaFace6不感兴趣~~~
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
	- #### 我要编译SeetaFace6  
	1. SeetaFace6官方编译方式
	参考：https://github.com/SeetaFace6Open/index#%E7%BC%96%E8%AF%91%E4%BE%9D%E8%B5%96
	2. 使用项目中的编译脚本
	参考：https://github.com/ViewFaceCore/ViewFaceCore/blob/main/docs/SeetaFace_Build.md
	
- ### **编译SeetaFaceBridge**

- ### **编译ViewFaceCore**

## 📄 文档
- [ViewFaceCore API](https://github.com/View12138/ViewFaceCore/blob/master/README_API.md)
- [*SeetaFace6 说明*](https://github.com/seetafaceengine/SeetaFace6/blob/master/README.md)
- [*SeetaFace 各接口说明*](https://github.com/seetafaceengine/SeetaFace6/tree/master/docs)
- [*SeetaFace 入门教程*](http://leanote.com/blog/post/5e7d6cecab64412ae60016ef)


## ❓ 常见问题
#### 1、Unable to load DLL 'ViewFaceBridge' or one of its dependencies  
1. 检查nuget包是否下载完全，编译目标文件夹下面的viewfacecore文件夹中是否有对应平台的依赖文件，比如说windows x64平台，在viewfacecore文件夹下面应该会有win/x64文件夹，文件夹中有很多*.dll文件。  
2. 缺少vc++依赖，安装nuget包`ViewFaceCore.runtime.win.vc`.[![](https://img.shields.io/nuget/v/ViewFaceCore.runtime.win.vc.svg)](https://www.nuget.org/packages/ViewFaceCore.runtime.win.vc)  

#### 2、开始人脸识别时卡死，然后异常结束，或者报异常：0x00007FFC3FDD104E (tennis.dll) (ConsoleApp1.exe 中)处有未经处理的异常: 0xC000001D: IllegInstruction。  
原因是tennis使用了不支持的指令集。下表是tennis文件对应支持的指令集。  
| 文件  | 指令集  | 说明  |
| ------------ | ------------ | ------------ |
| tennis.dll  | AVX2、FMA  | 默认  |
| tennis_haswell.dll  | AVX2、FMA   |   |
| tennis_sandy_bridge.dll  | AVX2   |   |
| tennis_pentium.dll  | SSE2   |   |

这个错误主要发生在低功耗CPU上面，低功耗CPU阉割了指令集。如果使用了不支持的指令集就会报这个异常。解决方案是删掉tennis.dll，然后用对应支持的指令集重命名为tennis.dll。比如在Intel奔腾低功耗CPU环境中，将tennis.dll删除，然后将tennis_pentium.dll重命名为tennis.dll。  


## 📦 使用许可   
<div align="center">

[Copyright (c) 2021, View](https://github.com/ViewFaceCore/ViewFaceCore/blob/main/LICENSE)
    |   
[*Copyright (c) 2019, SeetaTech*](https://github.com/SeetaFace6Open/index/blob/master/LICENSE)

</din>

> [\[源\]](https://github.com/SeetaFace6Open/index#%E8%81%94%E7%B3%BB%E6%88%91%E4%BB%AC) > *`SeetaFace` 开源版可以免费用于商业和个人用途。如果需要更多的商业支持，请联系商务邮件 bd@seetatech.com*

