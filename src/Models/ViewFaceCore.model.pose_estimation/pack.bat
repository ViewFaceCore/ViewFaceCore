@echo off

dotnet pack "ViewFaceCore.model.pose_estimation.csproj" -p:NuspecFile="ViewFaceCore.model.pose_estimation.nuspec" -c Release -o "..\ViewFaceCore.all_models\bin\Release\6.0.4"