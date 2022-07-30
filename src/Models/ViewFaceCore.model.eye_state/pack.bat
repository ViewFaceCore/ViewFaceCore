@echo off

:: 设置包生成配置
set configuration=%1
set version=%2
set output=%3

:: 获取当前目录名称作为包名
for %%a in ("%cd%") do set packageName=%%~nxa

dotnet pack "%packageName%.csproj" ^
    -p:NuspecFile="%packageName%.nuspec" ^
    -p:NuspecProperties="version=%version%" ^
    --configuration %configuration% ^
    --output "%output%\%configuration%\%version%"

:: 删除输出目录
rd /q /s "bin" "obj\"