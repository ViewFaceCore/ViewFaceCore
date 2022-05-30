@echo off

dotnet pack "ViewFaceCore.runtime.win.x86.csproj" -p:NuspecFile="ViewFaceCore.runtime.win.x86.nuspec" -c Release -o "..\ViewFaceCore.all_models\bin\Release\6.0.4"