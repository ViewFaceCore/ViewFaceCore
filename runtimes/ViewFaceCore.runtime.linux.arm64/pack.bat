@echo off

dotnet pack "ViewFaceCore.runtime.linux.arm64.csproj" -p:NuspecFile="ViewFaceCore.runtime.linux.arm64.nuspec" -c Release -o "..\ViewFaceCore.all_models\bin\Release\6.0.4"