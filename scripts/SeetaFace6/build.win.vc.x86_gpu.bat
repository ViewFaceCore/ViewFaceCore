@echo off

rem 设置控制台字符集为UTF-8
CHCP 65001 > nul
color 0f

rem 设置vcvarsall.bat位置
set VCVARSALL_DIR="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
rem 设置源码位置
set "BASE_DIR=%~dp0..\..\src\SeetaFace\index"
rem 设置编译结果位置
set "INSTALL_DIR=%~dp0..\..\src\SeetaFace\index\build"
rem 编译结果
set "BUILD_TYPE=Release"
rem set "BUILD_TYPE=Debug"
rem 编译架构
set "PLATFORM_TARGET=x86"
set "WITH_SSL=OFF"

rem 参数检查
if not exist %VCVARSALL_DIR% (
	color 0c
	echo vcvarsall.bat不存在，请指定vcvarsall.bat路径（修改脚本中的VCVARSALL_DIR为VS中vcvarsall.bat路径）
	pause > nul
	exit 0
)
if not exist %BASE_DIR% (
	color 0c
	echo 源码路径【%BASE_DIR%】不存在，请指定源码路径！
	pause > nul
	exit 0
)

rem 删除历史编译结果
if not exist %INSTALL_DIR% (
	md %INSTALL_DIR%
)
if exist "%INSTALL_DIR%\cmake" (
	rmdir /s /q "%INSTALL_DIR%\cmake"
)
if exist "%INSTALL_DIR%\include" (
	rmdir /s /q "%INSTALL_DIR%\include"
)
if exist "%INSTALL_DIR%\bin\%PLATFORM_TARGET%" (
	rmdir /s /q "%INSTALL_DIR%\bin\%PLATFORM_TARGET%"
)
if exist "%INSTALL_DIR%\lib\%PLATFORM_TARGET%" (
	rmdir /s /q "%INSTALL_DIR%\lib\%PLATFORM_TARGET%"
)

call %VCVARSALL_DIR% %PLATFORM_TARGET%
rem 编译三个前置模块
call :fun_build_target OpenRoleZoo
call :fun_build_seeta_authorize SeetaAuthorize
call :fun_build_tenniS TenniS
rem 编译其他模块
set buildList=FaceAntiSpoofingX6 FaceBoxes FaceRecognizer6 FaceTracker6 Landmarker PoseEstimator6 QualityAssessor3 SeetaAgePredictor SeetaEyeStateDetector SeetaGenderPredictor SeetaMaskDetector
(for %%w in (%buildList%) do ( 
	call :fun_build_target %%w
))

echo 编译结果：%BUILD_TYPE%
echo 编译架构：%PLATFORM_TARGET%
echo 编译目标：%INSTALL_DIR%
echo 使用CUDA：Yes
echo 编译结束，请按任意键退出...
pause > nul
exit 0

:fun_build_target
    echo Start build target %1
	set "BUILD_DIR=%BASE_DIR%\%1\craft\build.win.vc14.%PLATFORM_TARGET%"
	
	cd /d "%BASE_DIR%\%1\craft"
	if exist %BUILD_DIR% (
		rmdir /s /q %BUILD_DIR%
	)
	md %BUILD_DIR%
	cd /d %BUILD_DIR%
	
	cmake "%BASE_DIR%\%1" ^
		-G"NMake Makefiles JOM" ^
		-DCMAKE_BUILD_TYPE="%BUILD_TYPE%" ^
		-DCONFIGURATION="%BUILD_TYPE%" ^
		-DPLATFORM="%PLATFORM_TARGET%" ^
		-DORZ_ROOT_DIR="%INSTALL_DIR%" ^
		-DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ^
		-DCMAKE_MODULE_PATH="" ^
		-DSEETA_AUTHORIZE=OFF ^
		-DSEETA_MODEL_ENCRYPT=ON

	cmake --build . --target install
	cd /d %INSTALL_DIR%\lib\%PLATFORM_TARGET%"
	move /y *.dll "..\..\bin\%PLATFORM_TARGET%"
GOTO:EOF 

:fun_build_seeta_authorize
    echo Start build target %1
	set "BUILD_DIR=%BASE_DIR%\%1\craft\build.win.vc14.%PLATFORM_TARGET%"
	
	cd /d "%BASE_DIR%\%1\craft"
	if exist %BUILD_DIR% (
		rmdir /s /q %BUILD_DIR%
	)
	md %BUILD_DIR%
	cd /d %BUILD_DIR%
		
	cmake "%BASE_DIR%\%1" ^
		-G"NMake Makefiles JOM" ^
		-DCMAKE_BUILD_TYPE="%BUILD_TYPE%" ^
		-DPLATFORM="%PLATFORM_TARGET%" ^
		-DOPENSSL_ROOT_DIR="%SSL_HOME%" ^
		-DORZ_ROOT_DIR="%INSTALL_DIR%" ^
		-DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"

	cmake --build . --target install
	cd /d %BASE_DIR%\build\lib\%PLATFORM_TARGET%"
GOTO:EOF 

:fun_build_tenniS
    echo Start build target %1 with CUDA.
	set "BUILD_DIR=%BASE_DIR%\%1\craft\build.win.vc14.%PLATFORM_TARGET%_gpu"
	
	cd /d "%BASE_DIR%\%1\craft"
	if exist %BUILD_DIR% (
		rmdir /s /q %BUILD_DIR%
	)
	md %BUILD_DIR%
	cd /d %BUILD_DIR%
		
	cmake "%BASE_DIR%\%1" ^
		-G"NMake Makefiles JOM" ^
		-DCMAKE_BUILD_TYPE="%BUILD_TYPE%" ^
		-DCONFIGURATION="%BUILD_TYPE%" ^
		-DPLATFORM="%PLATFORM_TARGET%" ^
		-DORZ_ROOT_DIR="%INSTALL_DIR%" ^
		-DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ^
		-DTS_USE_CUDA=ON ^
		-DTS_USE_CUBLAS=ON ^
		-DTS_USE_OPENMP=ON ^
		-DTS_USE_SIMD=ON ^
		-DTS_ON_HASWELL=ON ^
		-DTS_DYNAMIC_INSTRUCTION=ON ^
		-DTS_BUILD_TEST=OFF ^
		-DTS_BUILD_TOOLS=OFF

	cmake --build . --target install
	cd /d %BASE_DIR%\build\lib\%PLATFORM_TARGET%"
GOTO:EOF 
