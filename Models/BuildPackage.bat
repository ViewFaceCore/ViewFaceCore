
cd ViewFaceCore.age_predictor
dotnet pack "ViewFaceCore.age_predictor.csproj" -p:NuspecFile="ViewFaceCore.age_predictor.nuspec" -c Release
cd ..

cd ViewFaceCore.eye_state
dotnet pack "ViewFaceCore.eye_state.csproj" -p:NuspecFile="ViewFaceCore.eye_state.nuspec" -c Release
cd ..

cd ViewFaceCore.face_detector
dotnet pack "ViewFaceCore.face_detector.csproj" -p:NuspecFile="ViewFaceCore.face_detector.nuspec" -c Release
cd ..

cd ViewFaceCore.face_landmarker_mask_pts5
dotnet pack "ViewFaceCore.face_landmarker_mask_pts5.csproj" -p:NuspecFile="ViewFaceCore.face_landmarker_mask_pts5.nuspec" -c Release
cd ..

cd ViewFaceCore.face_landmarker_pts5
dotnet pack "ViewFaceCore.face_landmarker_pts5.csproj" -p:NuspecFile="ViewFaceCore.face_landmarker_pts5.nuspec" -c Release
cd ..

cd ViewFaceCore.face_landmarker_pts68
dotnet pack "ViewFaceCore.face_landmarker_pts68.csproj" -p:NuspecFile="ViewFaceCore.face_landmarker_pts68.nuspec" -c Release
cd ..

cd ViewFaceCore.face_recognizer
dotnet pack "ViewFaceCore.face_recognizer.csproj" -p:NuspecFile="ViewFaceCore.face_recognizer.nuspec" -c Release
cd ..

cd ViewFaceCore.face_recognizer_light
dotnet pack "ViewFaceCore.face_recognizer_light.csproj" -p:NuspecFile="ViewFaceCore.face_recognizer_light.nuspec" -c Release
cd ..

cd ViewFaceCore.face_recognizer_mask
dotnet pack "ViewFaceCore.face_recognizer_mask.csproj" -p:NuspecFile="ViewFaceCore.face_recognizer_mask.nuspec" -c Release
cd ..

cd ViewFaceCore.fas_first
dotnet pack "ViewFaceCore.fas_first.csproj" -p:NuspecFile="ViewFaceCore.fas_first.nuspec" -c Release
cd ..

cd ViewFaceCore.fas_second
dotnet pack "ViewFaceCore.fas_second.csproj" -p:NuspecFile="ViewFaceCore.fas_second.nuspec" -c Release
cd ..

cd ViewFaceCore.gender_predictor
dotnet pack "ViewFaceCore.gender_predictor.csproj" -p:NuspecFile="ViewFaceCore.gender_predictor.nuspec" -c Release
cd ..

cd ViewFaceCore.mask_detector
dotnet pack "ViewFaceCore.mask_detector.csproj" -p:NuspecFile="ViewFaceCore.mask_detector.nuspec" -c Release
cd ..

cd ViewFaceCore.pose_estimation
dotnet pack "ViewFaceCore.pose_estimation.csproj" -p:NuspecFile="ViewFaceCore.pose_estimation.nuspec" -c Release
cd ..

cd ViewFaceCore.quality_lbn
dotnet pack "ViewFaceCore.quality_lbn.csproj" -p:NuspecFile="ViewFaceCore.quality_lbn.nuspec" -c Release
cd ..