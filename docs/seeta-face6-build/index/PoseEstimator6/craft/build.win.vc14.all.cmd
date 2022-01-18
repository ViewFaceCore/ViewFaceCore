@echo off

echo ^=^= building x64 ^=^=
call %~dp0build.win.vc14.x64.cmd

echo ^=^= building x64.debug ^=^=
call %~dp0build.win.vc14.x64.debug.cmd

echo ^=^= building x86 ^=^=
call %~dp0build.win.vc14.x86.cmd

echo ^=^= building x86.debug ^=^=
call %~dp0build.win.vc14.x86.debug.cmd