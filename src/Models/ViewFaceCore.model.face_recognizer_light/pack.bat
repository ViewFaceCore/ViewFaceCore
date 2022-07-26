@echo off

dotnet pack "ViewFaceCore.model.face_recognizer_light.csproj" -p:NuspecFile="ViewFaceCore.model.face_recognizer_light.nuspec" -c Release -o "..\ViewFaceCore.all_models\bin\Release\6.0.4"