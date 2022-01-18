@echo off

set "BUILD_DIR=build.win.vc14.x64.debug"
set "BUILD_TYPE=Debug"
set "PLATFORM=x64"
set "PLATFORM_TARGET=x64"

set "ORZ_HOME=../build"

set "INSTALL_DIR=../../../build"

call "%VctPath%\vcvarsall.bat" %PLATFORM%

cd %~dp0

md "%BUILD_DIR%"

cd "%BUILD_DIR%"

md "%INSTALL_DIR%"

cmake "%~dp0.." ^
-G"NMake Makefiles JOM" ^
-DCMAKE_BUILD_TYPE="%BUILD_TYPE%" ^
-DPLATFORM="%PLATFORM_TARGET%" ^
-DOPENSSL_ROOT_DIR="%SSL_HOME%" ^
-DORZ_ROOT_DIR="%ORZ_HOME%" ^
-DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"

jom -j16 install

exit /b

