@echo off

dotnet pack "ViewFaceCore.runtime.linux.arm.csproj" -p:NuspecFile="ViewFaceCore.runtime.linux.arm.nuspec" -c Release -o "..\ViewFaceCore.all_models\bin\Release\6.0.4"