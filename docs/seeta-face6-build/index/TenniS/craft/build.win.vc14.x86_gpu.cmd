@echo off

set "BUILD_DIR=build.win.vc14.x86_gpu"
set "BUILD_TYPE=Release"
set "PLATFORM=x86"
set "PLATFORM_TARGET=x86"

set "ORZ_HOME=D:\3rd\local"

set "INSTALL_DIR=%~dp0/../../build"

call "%VctPath%\vcvarsall.bat" %PLATFORM%

cd %~dp0

md "%BUILD_DIR%"

cd "%BUILD_DIR%"

md "%INSTALL_DIR%"

cmake "%~dp0.." ^
-G"NMake Makefiles JOM" ^
-DCMAKE_BUILD_TYPE="%BUILD_TYPE%" ^
-DCONFIGURATION="%BUILD_TYPE%" ^
-DPLATFORM="%PLATFORM_TARGET%" ^
-DORZ_ROOT_DIR="%ORZ_HOME%" ^
-DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ^
-DTS_USE_CUDA=ON ^
-DTS_USE_CUBLAS=ON ^
-DTS_USE_OPENMP=ON ^
-DTS_USE_SIMD=ON ^
-DTS_ON_HASWELL=ON

jom -j16 install

exit /b

