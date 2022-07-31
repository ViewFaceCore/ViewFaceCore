## ViewFaceCore

*一个基于 SeetaFace6 实现的 .NET 平台的人脸识别库。*

- ### `ViewFaceCore` 必须依赖某一个本机库  
  > - 针对不同的操作系统平台可以按照下表安装对应的包  

  | OS      | CPU   | Package Name                           | Version |
  |:--------|:------|:---------------------------------------|:--------:|
  | Windows | x64   | ViewFaceCore.runtime.win.x64           |  6.0.6  |
  | Windows | x86   | ViewFaceCore.runtime.win.x86           |  6.0.6  |
  | Ubuntu  | x64   | ViewFaceCore.runtime.ubuntu.20.04.x64  |  6.0.6  |
  | Linux   | arm   | ViewFaceCore.runtime.linux.arm         |  6.0.6  |
  | Linux   | arm64 | ViewFaceCore.runtime.linux.arm64       |  6.0.6  |

  > - 在 Windows 上由于没有 VC++ 运行时导致的问题可以尝试安装 `ViewFaceCore.runtime.win.vc` 包  

- ### `ViewFaceCore` 必须通过扩展包使用

  | Image Library  | Package Name                         | Version |
  |:--------------:|:-------------------------------------|:-------:|
  | System.Drawing | ViewFaceCore.Extension.SystemDrawing |  0.3.6  |
  | SkiaSharp      | ViewFaceCore.Extension.SkiaSharp     |  0.3.6  |

  > 如果你有其它的图形库需求, 请提交 Issue 或 PR

- ### `ViewFaceCore` 必须依赖某个模型包
  - 独立模型包 `ViewFaceCore.model.* ` 
    > 可以按照 `ViewFace` 的方法说明来添加需要模型包

  - 全部模型包 `ViewFaceCore.all_models`
    > 也可以直接使用此包以添加所有模型包