@echo off

set configuration=Release
set version=6.0.6
set output=..\packages

echo ====================== pack ViewFaceCore.model.age_predictor ======================
cd ViewFaceCore.model.age_predictor
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.model.eye_state ======================
cd ViewFaceCore.model.eye_state
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.model.face_detector ======================
cd ViewFaceCore.model.face_detector
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.model.face_landmarker_mask_pts5 ======================
cd ViewFaceCore.model.face_landmarker_mask_pts5
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.model.face_landmarker_pts5 ======================
cd ViewFaceCore.model.face_landmarker_pts5
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.model.face_landmarker_pts68 ======================
cd ViewFaceCore.model.face_landmarker_pts68
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.model.face_recognizer ======================
cd ViewFaceCore.model.face_recognizer
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.model.face_recognizer_light ======================
cd ViewFaceCore.model.face_recognizer_light
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.model.face_recognizer_mask ======================
cd ViewFaceCore.model.face_recognizer_mask
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.model.fas_first ======================
cd ViewFaceCore.model.fas_first
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.model.fas_second ======================
cd ViewFaceCore.model.fas_second
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.model.gender_predictor ======================
cd ViewFaceCore.model.gender_predictor
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.model.mask_detector ======================
cd ViewFaceCore.model.mask_detector
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.model.pose_estimation ======================
cd ViewFaceCore.model.pose_estimation
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.model.quality_lbn ======================
cd ViewFaceCore.model.quality_lbn
call pack.bat %configuration% %version% %output%
cd ..

echo ====================== pack ViewFaceCore.all_models ======================
cd ViewFaceCore.all_models
call pack.bat %configuration% %version% %output%
cd ..