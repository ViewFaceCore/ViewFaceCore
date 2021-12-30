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
   | .NET / .NET Core |3.1 | win ( x64/x86 )、linux ( arm/arm64/x64 ) |

- ### 简单的人脸信息检测  
   - 以 Windows x64 为例
   1. 使用 [nuget](https://www.nuget.org) 安装依赖
      | 包名称 | 最小版本 | 生成文件夹 | 说明 |
      | :- | :-: | - | - |
      | [ViewFaceCore](https://www.nuget.org/packages/ViewFaceCore/) | `0.3.5` | —— | ViewFaceCore .NET 核心库 |
      | [ViewFaceCore.model.face_detector](https://www.nuget.org/packages/ViewFaceCore.model.face_detector) | `6.0.0` | `models` | 人脸检测的模型支持 |
      | [ViewFaceCore.runtime.win.x64](https://www.nuget.org/packages/ViewFaceCore.runtime.win.x64) | `6.0.2` | `viewfacecore\win\x64` | Windows-x64 的本机运行时 |

   2. 获取人脸信息  
      ```csharp
      using System;
      using System.Drawing;
      using ViewFaceCore.Sharp;
      
      namespace YourFaceProject
      {
          class Program
          {
              static void Main(string[] args)
              {
                  ViewFace face = new ViewFace();
                  string filename = @"[your face image file path]";
                  Bitmap bitmap = (Bitmap)Image.FromFile(filename);
                  var infos = face.FaceDetector(bitmap);
                  Console.WriteLine($"识别到的人脸数量：{infos.Length} 。人脸信息：\n");
                  Console.WriteLine($"No.\t人脸置信度\t位置信息");
                  for (int i = 0; i < infos.Length; i++)
                  {
                      Console.WriteLine($"{i}\t{infos[i].Score:f8}\t{infos[i].Location}");
                  }
                  Console.Read();
              }
          }
      }
      ```





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

   `全部重新编译`  
   1. 配置 %VctPath% 环境变量 (即：vcvarsall.bat 脚本的路径)
      > 以 Visual Studio 2022 为例：  
      > `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build`
   2. 按照 [SeetaFace 编译依赖](https://github.com/SeetaFace6Open/index#%E7%BC%96%E8%AF%91%E4%BE%9D%E8%B5%96) 配置好依赖工具
      - 本项目使用 MSVC (win) / GCC (linux on wsl) 编译完成
      > 1. 编译工具
      > 2. For linux<br>
      >  GNU Make 工具<br>
      >  GCC 或者 Clang 编译器
      > 3. For windows<br>
      >  [MSVC](https://visualstudio.microsoft.com/zh-hans/) 或者 MinGW. <br>
      >  [jom](https://wiki.qt.io/Jom)
      > 4. [CMake](http://www.cmake.org/)
      > 5. 依赖架构<br>
      >  CPU 支持 AVX 和 FMA [可选]（x86）或 NENO（ARM）支持
   3. 首先编译 `OpenRoleZoo `、`SeetaAuthorize`、`TenniS` 三个项目
      - 在项目的 `craft` 文件夹下启动 shell
      > **`powershell`** > `./build.win.vc14.all.cmd`  
      > **`linux shell(wsl)`** > `./build.linux.all.sh`
   4. 然后编译其他项目 `SeetaMaskDetector`、`FaceAntiSpoofingX6`、`FaceBoxes`、`FaceRecognizer6`、`FaceTracker6`、`Landmarker`、`OpenRoleZoo`、`PoseEstimator6`、`QualityAssessor3`、`SeetaAgePredictor`、`SeetaAuthorize`、`SeetaEyeStateDetector`、`SeetaGenderPredictor`  
      - 在项目的 `craft` 文件夹下启动 shell
      > **`powershell`** > `./build.win.vc14.all.cmd`  
      > **`linux shell(wsl)`** > `./build.linux.all.sh`


## 📄&nbsp;文档
- [ViewFaceCore API](https://github.com/View12138/ViewFaceCore/blob/master/README_API.md)
- [SeetaFace6 说明](https://github.com/seetafaceengine/SeetaFace6/blob/master/README.md)
- [SeetaFace 入门教程](http://leanote.com/blog/post/5e7d6cecab64412ae60016ef)
- [SeetaFace 各接口说明](https://github.com/seetafaceengine/SeetaFace6/tree/master/docs)


## ❓&nbsp;常见问题

## 📦&nbsp;使用许可
<div align="center">

[MIT](https://github.com/ViewFaceCore/ViewFaceCore/blob/main/LICENSE) © 2021 View

</din>
