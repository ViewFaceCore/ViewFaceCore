@echo off

dotnet pack "ViewFaceCore.runtime.win.x64.csproj" -p:NuspecFile="ViewFaceCore.runtime.win.x64.nuspec" -c Release -o "..\ViewFaceCore.all_models\bin\Release\6.0.4"