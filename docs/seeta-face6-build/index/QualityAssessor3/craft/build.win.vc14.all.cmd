@echo off

echo ^=^= building x64 ^=^=
call %~dp0build.win.vc14.x64.cmd

echo ^=^= building x64.debug ^=^=
call %~dp0build.win.vc14.x64.debug.cmd

echo ^=^= building x86 ^=^=
call %~dp0build.win.vc14.x86.cmd

echo ^=^= building x86.debug ^=^=
call %~dp0build.win.vc14.x86.debug.cmd

move "../../../build/lib/x64/SeetaQualityAssessor300.dll" "../../../build//bin/x64/"
move "../../../build/lib/x64/SeetaQualityAssessor300d.dll" "../../../build//bin/x64/"
move "../../../build/lib/x86/SeetaQualityAssessor300.dll" "../../../build//bin/x86/"
move "../../../build/lib/x86/SeetaQualityAssessor300d.dll" "../../../build//bin/x86/"