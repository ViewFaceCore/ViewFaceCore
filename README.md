<style>
table th:first-of-type {
    min-width: 350px
}
table th:nth-of-type(2) {
    min-width: 64px
}
table th:nth-of-type(3) {
    width: 100%;
}
</style>

<div align="center">

# ViewFaceCore 
[![Nuget](https://img.shields.io/nuget/v/ViewFaceCore?color=%233F48CC&style=flat-square)](https://www.nuget.org/packages/ViewFaceCore/) &nbsp;&nbsp;
[![GitHub license](https://img.shields.io/github/license/ViewFaceCore/ViewFaceCore?style=flat-square)](https://github.com/ViewFaceCore/ViewFaceCore/blob/main/LICENSE) &nbsp;&nbsp;
![GitHub stars](https://img.shields.io/github/stars/ViewFaceCore/ViewFaceCore?color=%23FCD53F&style=flat-square) &nbsp;&nbsp;
![GitHub forks](https://img.shields.io/github/forks/ViewFaceCore/ViewFaceCore?style=flat-square)

<br/>

—— [💎 关于](#💎-关于) &nbsp;| [⭐ 快速开始](#⭐-快速开始) &nbsp;| [🔧 构建](#🔧-构建) &nbsp;| [📦 包](#📦-包) &nbsp;| [🐟 API](#🐟-api) &nbsp; ——
<br/>
—— [🔎 参考](#🔎-参考) &nbsp;| [❓ 问答](#❓-问答) &nbsp;| [🧩 贡献](#🧩-贡献) &nbsp;| [📄 许可](#📄-许可) &nbsp; ——

</div>

## 💎 关于
- 一个基于 [SeetaFace6](https://github.com/SeetaFace6Open/index) 的 .NET 人脸识别解决方案
- 本项目受到了 [SeetaFaceEngine.Net](https://github.com/iarray/SeetaFaceEngine.Net) 的启发
- 开源、免费、跨平台

> 受支持的 .NET 框架 和 操作系统  

|   Target Framework    | ViewFaceCore Version |             operating system             |
|:---------------------:|:--------------------:|:----------------------------------------:|
| .NET Framework >= 4.0 |        0.3.7         |            windows (x64/x86)             |
| .NET Standard >= 2.0  |        0.3.7         |            windows (x64/x86)             |
|   .NET Core >= 3.1    |        0.3.7         | windows (x64/x86), linux (arm/arm64/x64) |
|      .NET >= 5.0      |        latest        | windows (x64/x86), linux (arm/arm64/x64) |


## ⭐ 快速开始

- [Examples](/src/Examples/)  

- 在 *Windows x64* 下, 快速集成人脸检测  

  1. 创建 `.net6` 控制台项目
  1. 使用 [Nuget](https://www.nuget.org) 安装以下依赖
  
     | 包名称                                                                                                      | 版本                                                                                                                            | 说明                        |
     |:------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|-----------------------------|
     | [ViewFaceCore](https://www.nuget.org/packages/ViewFaceCore/)                                                | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.svg?color=%233F48CC&label=%20&style=flat-square)                         | *ViewFaceCore* 核心包       |
     | [ViewFaceCore.model.face_detector](https://www.nuget.org/packages/ViewFaceCore.model.face_detector)         | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.model.face_detector.svg?color=%233F48CC&label=%20&style=flat-square)     | *人脸检测* 模型包           |
     | [ViewFaceCore.runtime.win.x64](https://www.nuget.org/packages/ViewFaceCore.runtime.win.x64)                 | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.runtime.win.x64.svg?color=%233F48CC&label=%20&style=flat-square)         | *Windows-x64* 运行时包      |
     | [ViewFaceCore.Extension.SystemDrawing](https://www.nuget.org/packages/ViewFaceCore.Extension.SystemDrawing) | ![Version](https://img.shields.io/nuget/v/ViewFaceCore.Extension.SystemDrawing.svg?color=%233F48CC&label=%20&style=flat-square) | *System.Drawing* 图像扩展包 |
  
  1. 获取人脸信息
  
  ```csharp
  using System;
  using System.Drawing;
  using ViewFaceCore.Core;
  using ViewFaceCore.Model;
  
  namespace ViewFaceCore.Example.ConsoleApp;
  
  internal class Program
  {
      static void Main(string[] args)
      {
          string imagePath = @"images/Jay_3.jpg";
          using var bitmap = (Bitmap)Image.FromFile(imagePath);
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
  
  ```

## 🔧 构建
   
- [*SeetaFace6 构建*](/docs/SeetaFace_Build.md)
- [*ViewFaceCore 构建*](/docs/ViewFaceCore_Build.md)

## 📦 包

- [*ViewFaceCore 的 Nuget 包清单*](/docs/Packages.md)

## 🐟 API

- [*ViewFaceCore API*](/docs/ViewFaceCore_API.md)

## 🔎 参考
- [*SeetaFace6 说明*](https://github.com/seetafaceengine/SeetaFace6/blob/master/README.md)
- [*SeetaFace 各接口说明*](https://github.com/seetafaceengine/SeetaFace6/tree/master/docs)
- [*SeetaFace 入门教程*](http://leanote.com/blog/post/5e7d6cecab64412ae60016ef)


## ❓ 问答

- [Issues](https://github.com/ViewFaceCore/ViewFaceCore/issues)
- [常见问题](/docs/QA.md)

## 🧩 贡献

- [PR](https://github.com/ViewFaceCore/ViewFaceCore/pull)
- [参与贡献](/docs/Contribute.md)

## 📄 许可   
<div align="center">

[Copyright (c) 2021, View](https://github.com/ViewFaceCore/ViewFaceCore/blob/main/LICENSE) | [*Copyright (c) 2019, SeetaTech*](https://github.com/SeetaFace6Open/index/blob/master/LICENSE)

</div>

> *[SeetaFace 开源版](https://github.com/SeetaFace6Open/index#%E8%81%94%E7%B3%BB%E6%88%91%E4%BB%AC) 可以免费用于商业和个人用途。如果需要更多的商业支持，请联系商务邮件 bd@seetatech.com*
