# ViewFaceCore
- 示例程序 : [WinForm 摄像头人脸检测](https://github.com/View12138/ViewFaceCoreDemo)
- 已发布至 : [Nuget.org](https://www.nuget.org/packages/ViewFaceCore/)
- 当前版本 : **0.3.4**

### ⭐、关于
  - 这是一个使用超简单的 C# 人脸识别库。
  - 此包是开源的、且免费、无限制的提供你使用，或者用于商业行为。
  - 如果你觉得此项目还不错，可以请作者喝果汁，或者买瓶霸王洗发水。
     - <img src="https://i.loli.net/2020/09/11/N5ifZezGTPxCVBc.jpg" height="300px"/> <img src="https://i.loli.net/2020/09/11/P4CAegTjbvG3tr7.png" height="300px"/>

### 📘、API 文档
[查看 API 文档](https://github.com/View12138/ViewFaceCore/blob/master/README_API.md)

### 一、使用

1. 创建你的 .NET 应用，并且你的 .NET 版本需要满足以下条件：
   - .NET Standard >= 2.0
   - .NET Core >= 2.0
   - .NET Framework >= 4.6.1^2

2. 使用 Nuget 安装 **`ViewFaceCore`**
   - Author : *View*
   - Version : *Latest*
   > 此 Nuget 包会自动添加依赖的 C++ 库，以及最精简的识别模型。(`face_detector.csta`)  
   > 请自行下载需要的 [SeetaFace6 模型文件](https://github.com/seetafaceengine/SeetaFace6#%E7%99%BE%E5%BA%A6%E7%BD%91%E7%9B%98)。  
   > 若没有硬盘要求，建议下载全部模型。

3. 在项目中编写你的代码
   - 一个简单的例子 `ViewFaceTest/Program.cs`

4. 构建  
   1. 生成你的项目，此时项目的生成目录中会出现 `model` 文件夹。
   2. 将下载的 ****.csta*** 模型文件拷贝至 `model` 文件夹。  
      - 也可以使用 生成命令自动复制模型文件至输出目录

### 二、项目说明

| 项目 | 语言 | 说明 |
| - | - | - |
| ViewFace | `C++` | 基于 `SeetaFace6` 的接口封装，支持 x86、x64 |
| ViewFaceCore | `C#` | 基于 `ViewFace` 的 C# 形式的封装，支持 AnyCPU |
| ViewFaceTest | `C#` | 调用 `ViewFaceCore` 实现的简单的图片人脸识别 |
| ViewFaceTestPackage | `C#` | 调用 Nuget 中的 `ViewFaceCore` 包 实现的简单的图片人脸识别 |

### 三、编译本项目
1. 开发环境：  
   - 开发工具 : Visual Studio 2019 16.7.1
   - 操作系统 : Windows 10 专业版 2004 19041.450
2. 依赖：
   - 下载 [SeetaFace6 开发包](https://github.com/seetafaceengine/SeetaFace6#%E7%99%BE%E5%BA%A6%E7%BD%91%E7%9B%98)
   - SeetaFace 开发包头文件存放路径 : `C:\vclib\seeta\include\seeta`
   - SeetaFace 开发包的 x86 和 x64 的类库的存放路径 : `C:\vclib\seeta\lib`
3. 编译流程 (Release) ：
   1. 分别编译 x86 和 x64 模式的 `ViewFace` 项目。
   2. 切换到 AnyCPU ，并编译 `ViewFaceCore` 项目。
   > 或者使用 ReBuild.bat 自动编译。
