@echo off

dotnet pack "ViewFaceCore.runtime.ubuntu.20.04.x64.csproj" -p:NuspecFile="ViewFaceCore.runtime.ubuntu.20.04.x64.nuspec" -c Release -o "..\ViewFaceCore.all_models\bin\Release\6.0.4"