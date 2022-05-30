@echo off

dotnet pack "ViewFaceCore.model.mask_detector.csproj" -p:NuspecFile="ViewFaceCore.model.mask_detector.nuspec" -c Release -o "..\ViewFaceCore.all_models\bin\Release\6.0.4"