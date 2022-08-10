<div align="center">

# ViewFaceCore 
[![Nuget](https://img.shields.io/nuget/v/ViewFaceCore)](https://www.nuget.org/packages/ViewFaceCore/) &nbsp;&nbsp;
[![GitHub license](https://img.shields.io/github/license/ViewFaceCore/ViewFaceCore)](https://github.com/ViewFaceCore/ViewFaceCore/blob/main/LICENSE) &nbsp;&nbsp;
![GitHub stars](https://img.shields.io/github/stars/ViewFaceCore/ViewFaceCore?style=flat) &nbsp;&nbsp;
![GitHub forks](https://img.shields.io/github/forks/ViewFaceCore/ViewFaceCore)

<br/>

—— [🪧 关于](#🪧&nbsp;关于) &nbsp;| [⭐ 快速开始](#⭐&nbsp;快速开始) &nbsp;| [🔧 构建](#🔧&nbsp;构建) &nbsp;| [📄 文档](#📄&nbsp;文档) &nbsp;| [❓ 常见问题](#❓&nbsp;常见问题) &nbsp;| [📦 使用许可](#📦&nbsp;使用许可) ——

</div>

## 🪧&nbsp;关于
- 一个基于 [SeetaFace6](https://github.com/SeetaFace6Open/index) 的 .NET 人脸识别解决方案
- 本项目受到了 [SeetaFaceEngine.Net](https://github.com/iarray/SeetaFaceEngine.Net) 的启发
- 开源、免费、跨平台 (win/linux)

## ⭐&nbsp;快速开始
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
      | [ViewFaceCore](https://www.nuget.org/packages/ViewFaceCore/) | `0.3.6` | —— | ViewFaceCore .NET 核心库 |
      | [ViewFaceCore.all_models](https://www.nuget.org/packages/ViewFaceCore.all_models) | `6.0.6` | `viewfacecore\models` | 人脸检测的模型支持(图省事可以直接安装这个) |
      | [ViewFaceCore.runtime.win.x64](https://www.nuget.org/packages/ViewFaceCore.runtime.win.x64) | `6.0.6` | `viewfacecore\win\x64` | Windows-x64 的本机运行时，其它平台自行选择安装，可安装多个 |
	  | [ViewFaceCore.Extension.SkiaSharp](https://www.nuget.org/packages/ViewFaceCore.Extension.SkiaSharp) | `6.0.6` |  —— | SkiaSharp图像处理扩展，ImageSharp、SkiaSharp、System.Drawing三选一 |

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


## 🔧&nbsp;构建
- ### **项目结构**

  Bridges  
  Models  
  SeetaFace/index  
  Simples  
  Tests  
  ViewFaceCore  

- ### **开发环境**
   - Visual Studio 2022 (17.0.2)
   - Windows 11 专业版 (21H2)
   - Ubuntu 20.04 (WSL)

- ### **编译过程**

   `使用` [SeetaFace6 开发包](https://github.com/seetafaceengine/SeetaFace6#%E7%99%BE%E5%BA%A6%E7%BD%91%E7%9B%98) `编译`
   | 描述 | 后缀名 | 放置路径 |
   | - | - | - |
   | 头文件 | *.h | `ViewFaceCore\SeetaFace\index\build\include\seeta\` |
   | —— | —— | —— |
   | Windows 开发包 (x64) | *.dll | `ViewFaceCore\SeetaFace\index\build\bin\x64\` |
   | Windows 开发包 (x64) | *.lib | `ViewFaceCore\SeetaFace\index\build\lib\x64\` |
   ||||
   | Windows 开发包 (x86) | *.dll | `ViewFaceCore\SeetaFace\index\build\bin\x86\` |
   | Windows 开发包 (x86) | *.lib | `ViewFaceCore\SeetaFace\index\build\lib\x86\` |
   ||||
   | Ubuntu 开发包 (x64) | *.so | `ViewFaceCore\SeetaFace\index\build\lib64\` |
   | Ubuntu 开发包 (arm64) | *.so | `ViewFaceCore\SeetaFace\index\build\lib\arm64\` |
   | Ubuntu 开发包 (arm) | *.so | `ViewFaceCore\SeetaFace\index\build\lib\arm\` |

   参考：https://github.com/ViewFaceCore/ViewFaceCore/blob/dev_20220725/docs/SeetaFace_Build.md


## 📄&nbsp;文档
- [ViewFaceCore API](https://github.com/View12138/ViewFaceCore/blob/master/README_API.md)
- [*SeetaFace6 说明*](https://github.com/seetafaceengine/SeetaFace6/blob/master/README.md)
- [*SeetaFace 各接口说明*](https://github.com/seetafaceengine/SeetaFace6/tree/master/docs)
- [*SeetaFace 入门教程*](http://leanote.com/blog/post/5e7d6cecab64412ae60016ef)


## ❓&nbsp;常见问题

## 📦&nbsp;使用许可   
<div align="center">

[Copyright (c) 2021, View](https://github.com/ViewFaceCore/ViewFaceCore/blob/main/LICENSE)
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
[*Copyright (c) 2019, SeetaTech*](https://github.com/SeetaFace6Open/index/blob/master/LICENSE)

</din>

> [\[源\]](https://github.com/SeetaFace6Open/index#%E8%81%94%E7%B3%BB%E6%88%91%E4%BB%AC) > *`SeetaFace` 开源版可以免费用于商业和个人用途。如果需要更多的商业支持，请联系商务邮件 bd@seetatech.com*

