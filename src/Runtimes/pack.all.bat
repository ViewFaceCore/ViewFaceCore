@echo off

set configuration=Release
set version=6.0.8-alpha3
set output=..\packages

echo ====================== pack ViewFaceCore.runtime.win.x64 ======================
cd ViewFaceCore.runtime.win.x64
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.runtime.win.x86 ======================
cd ViewFaceCore.runtime.win.x86
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.runtime.ubuntu.20.04.x64 ======================
cd ViewFaceCore.runtime.ubuntu.20.04.x64
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.runtime.linux.arm ======================
cd ViewFaceCore.runtime.linux.arm
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.runtime.linux.arm64 ======================
cd ViewFaceCore.runtime.linux.arm64
call pack.bat %configuration% %version% %output%
cd ..

echo 发布完成，请按任意键退出...
pause > nul
exit 0