@echo off

echo ^=^= building x64 ^=^=
call %~dp0build.win.vc14.x64.cmd

echo ^=^= building x86 ^=^=
call %~dp0build.win.vc14.x86.cmd