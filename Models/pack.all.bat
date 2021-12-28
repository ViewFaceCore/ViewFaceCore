@echo off

echo ====================== pack ViewFaceCore.age_predictor ======================
cd ViewFaceCore.age_predictor
call pack.bat
cd ..

echo ====================== pack ViewFaceCore.eye_state ======================
cd ViewFaceCore.eye_state
call pack.bat
cd ..

echo ====================== pack ViewFaceCore.face_detector ======================
cd ViewFaceCore.face_detector
call pack.bat
cd ..

echo ====================== pack ViewFaceCore.face_landmarker_mask_pts5 ======================
cd ViewFaceCore.face_landmarker_mask_pts5
call pack.bat
cd ..

echo ====================== pack ViewFaceCore.face_landmarker_pts5 ======================
cd ViewFaceCore.face_landmarker_pts5
call pack.bat
cd ..

echo ====================== pack ViewFaceCore.face_landmarker_pts68 ======================
cd ViewFaceCore.face_landmarker_pts68
call pack.bat
cd ..

echo ====================== pack ViewFaceCore.face_recognizer ======================
cd ViewFaceCore.face_recognizer
call pack.bat
cd ..

echo ====================== pack ViewFaceCore.face_recognizer_light ======================
cd ViewFaceCore.face_recognizer_light
call pack.bat
cd ..

echo ====================== pack ViewFaceCore.face_recognizer_mask ======================
cd ViewFaceCore.face_recognizer_mask
call pack.bat
cd ..

echo ====================== pack ViewFaceCore.fas_first ======================
cd ViewFaceCore.fas_first
call pack.bat
cd ..

echo ====================== pack ViewFaceCore.fas_second ======================
cd ViewFaceCore.fas_second
call pack.bat
cd ..

echo ====================== pack ViewFaceCore.gender_predictor ======================
cd ViewFaceCore.gender_predictor
call pack.bat
cd ..

echo ====================== pack ViewFaceCore.mask_detector ======================
cd ViewFaceCore.mask_detector
call pack.bat
cd ..

echo ====================== pack ViewFaceCore.pose_estimation ======================
cd ViewFaceCore.pose_estimation
call pack.bat
cd ..

echo ====================== pack ViewFaceCore.quality_lbn ======================
cd ViewFaceCore.quality_lbn
call pack.bat
cd ..

echo ====================== pack ViewFaceCore.all_models ======================
cd ViewFaceCore.all_models
dotnet pack "ViewFaceCore.all_models.csproj" -c Release
cd ..