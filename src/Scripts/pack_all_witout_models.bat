@echo off

set configuration=Release
set runTimeVersion=6.0.8-alpha3
set extensionsVersion=0.4.0-alpha3
set mainVersion=0.4.0-alpha3
set output=%~dp0publish

cd ..

echo 开始打包Runtimes...

echo ====================== pack ViewFaceCore.runtime.win.x64 ======================
cd .\Runtimes\ViewFaceCore.runtime.win.x64
call pack.bat %configuration% %runTimeVersion% %output%
cd ..\..

echo ====================== pack ViewFaceCore.runtime.win.x86 ======================
cd .\Runtimes\ViewFaceCore.runtime.win.x86
call pack.bat %configuration% %runTimeVersion% %output%
cd ..\..

echo ====================== pack ViewFaceCore.runtime.ubuntu.20.04.x64 ======================
cd .\Runtimes\ViewFaceCore.runtime.ubuntu.20.04.x64
call pack.bat %configuration% %runTimeVersion% %output%
cd ..\..

echo ====================== pack ViewFaceCore.runtime.linux.arm ======================
cd .\Runtimes\ViewFaceCore.runtime.linux.arm
call pack.bat %configuration% %runTimeVersion% %output%
cd ..\..

echo ====================== pack ViewFaceCore.runtime.linux.arm64 ======================
cd .\Runtimes\ViewFaceCore.runtime.linux.arm64
call pack.bat %configuration% %runTimeVersion% %output%
cd ..\..

echo 开始打包Extensions...

echo ====================== pack ViewFaceCore.Extension.DependencyInjection ======================
cd .\Extensions\ViewFaceCore.Extension.DependencyInjection
call pack.bat %configuration% %extensionsVersion% %output%
cd ..\..

echo ====================== pack ViewFaceCore.Extension.ImageSharp ======================
cd .\Extensions\ViewFaceCore.Extension.ImageSharp
call pack.bat %configuration% %extensionsVersion% %output%
cd ..\..

echo ====================== pack ViewFaceCore.Extension.SkiaSharp ======================
cd .\Extensions\ViewFaceCore.Extension.SkiaSharp
call pack.bat %configuration% %extensionsVersion% %output%
cd ..\..

echo ====================== pack ViewFaceCore.Extension.SystemDrawing ======================
cd .\Extensions\ViewFaceCore.Extension.SystemDrawing
call pack.bat %configuration% %extensionsVersion% %output%
cd ..\..

echo 开始打包ViewFaceCore...

echo ====================== pack ViewFaceCore ======================
cd .\ViewFaceCore
call pack.bat %configuration% %mainVersion% %output%
cd ..

echo 发布完成，请按任意键退出...
pause > nul
exit 0