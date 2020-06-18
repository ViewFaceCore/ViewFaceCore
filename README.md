# ViewFaceCore
超简单的人脸识别库。

# 使用

### 一、1 分钟在你的项目中集成人脸识别

#### 1. 创建你的 .NET 应用
  - .NET Standard >= 2.0
  - .NET Core >= 2.0
  - .NET Framework >= 4.6.1^2
  
#### 2. 使用 Nuget 安装 `ViewFaceCore`
  - Author : View
  - Version >= 0.1.1

#### 3. 在项目中编写你的代码
  - 一个简单的例子 `ViewFaceTest/Program.cs`

### 二、方法说明

```
using System.Drawing;
using ViewFaceCore.Sharp;
using ViewFaceCore.Sharp.Model;

// 识别 bitmap 中的人脸，并返回人脸的信息。
FaceInfo[] FaceDetector(Bitmap bitmap);

// 识别 bitmap 中指定的人脸信息 info 的关键点坐标。
FaceMarkPoint[] FaceMark(Bitmap bitmap, FaceInfo info)

```

# 说明

# 依赖
