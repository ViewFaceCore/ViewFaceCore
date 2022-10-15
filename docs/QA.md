# 常见问题

1. Error: `DirectoryNotFoundException: Can not found library path.`

   - 请检查对应目录下是否存在 Runtime 依赖, 有时网络问题会导致 Nuget 包下载失败

  
1. Unable to load DLL 'ViewFaceBridge' or one of its dependencies

	- 检查nuget包是否下载完全, 编译目标文件夹下面的viewfacecore文件夹中是否有对应平台的依赖文件, 比如说windows x64平台, 在viewfacecore文件夹下面应该会有win/x64文件夹, 文件夹中有很多*.dll文件。  
	- 缺少vc++依赖, 安装 Nuge t包 [ViewFaceCore.runtime.win.vc](https://www.nuget.org/packages/ViewFaceCore.runtime.win.vc)

1. 开始人脸识别时卡死, 然后异常结束
   > 或者报异常：0x00007FFC3FDD104E (tennis.dll) (ConsoleApp1.exe 中)处有未经处理的异常: 0xC000001D: IllegInstruction

	- 参考：[特定指令集支持](/docs/README_API.md#特定指令集支持)
