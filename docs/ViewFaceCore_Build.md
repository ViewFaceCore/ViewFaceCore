# ViewFaceCore 构建

## 项目结构

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

## 开发环境
   - Visual Studio 2022
   - Windows 10/11
   - Ubuntu 20.04 (WSL)、Debian 10/11等