@echo off

dotnet pack "ViewFaceCore.model.face_detector.csproj" -p:NuspecFile="ViewFaceCore.model.face_detector.nuspec" -c Release -o "..\ViewFaceCore.all_models\bin\Release\6.0.4"