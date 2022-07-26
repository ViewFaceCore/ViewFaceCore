@echo off

dotnet pack "ViewFaceCore.model.age_predictor.csproj" -p:NuspecFile="ViewFaceCore.model.age_predictor.nuspec" -c Release -o "..\ViewFaceCore.all_models\bin\Release\6.0.4"