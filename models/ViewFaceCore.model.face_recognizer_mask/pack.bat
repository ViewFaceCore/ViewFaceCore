@echo off

dotnet pack "ViewFaceCore.model.face_recognizer_mask.csproj" -p:NuspecFile="ViewFaceCore.model.face_recognizer_mask.nuspec" -c Release -o "..\ViewFaceCore.all_models\bin\Release\6.0.4"