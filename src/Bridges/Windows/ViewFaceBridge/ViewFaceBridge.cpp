#include "environment.h"

#ifndef STDCALL
#define STDCALL 
#endif // !STDCALL

#ifndef View_Api
#define View_Api 
#endif // !View_Api


#include "seeta/FaceDetector.h"
#include "seeta/FaceLandmarker.h"
#include "seeta/FaceRecognizer.h"
#include "seeta/FaceAntiSpoofing.h"
#include "seeta/FaceTracker.h"

#include "seeta/QualityOfBrightness.h"
#include "seeta/QualityOfClarity.h"
#include "seeta/QualityOfIntegrity.h"
#include "seeta/QualityOfPose.h"
#include "seeta/QualityOfPoseEx.h"
#include "seeta/QualityOfResolution.h"

#include "seetaEx/QualityOfClarityEx.h"
#include "seetaEx/QualityOfNoMask.h"

#include "seeta/AgePredictor.h"
#include "seeta/GenderPredictor.h"
#include "seeta/EyeStateDetector.h"

#include <iostream>
#include <string>

using namespace std;
using namespace seeta;

string modelPath = "./viewfacecore/models/"; // 模型所在路径
// 设置人脸模型目录
View_Api void SetModelPath(const char* path)
{
	modelPath = path;
}

// 获取人脸模型目录
View_Api void GetModelPath(char** path)
{
	strcpy(*path, modelPath.c_str());
}

// 释放由 malloc 分配的内存
View_Api void Free(void* address) {
	free(address);
}

// 获取人脸数量
View_Api SeetaFaceInfo* Detector(SeetaImageData& img, int* size, double faceSize = 20, double threshold = 0.9, double maxWidth = 2000, double maxHeight = 2000, int type = 0)
{
	FaceDetector faceDetector(ModelSetting(modelPath + (type == 0 ? "face_detector.csta" : "mask_detector.csta")));
	faceDetector.set(FaceDetector::Property::PROPERTY_MIN_FACE_SIZE, faceSize);
	faceDetector.set(FaceDetector::Property::PROPERTY_THRESHOLD, threshold);
	faceDetector.set(FaceDetector::Property::PROPERTY_MAX_IMAGE_WIDTH, maxWidth);
	faceDetector.set(FaceDetector::Property::PROPERTY_MAX_IMAGE_HEIGHT, maxHeight);

	auto faces = faceDetector.detect_v2(img);

	*size = faces.size();
	SeetaFaceInfo* _infos = (SeetaFaceInfo*)malloc((*size) * sizeof(SeetaFaceInfo));
	int index = 0;
	for (auto& face : faces)
	{
		_infos[index] = face;
		index++;
	}
	return _infos;
}

// 人脸关键点器
View_Api SeetaPointF* FaceMark(SeetaImageData& img, SeetaRect faceRect, long* size, int type = 0)
{
	string modelName = "face_landmarker_pts68.csta";
	if (type == 1) { modelName = "face_landmarker_mask_pts5.csta"; }
	if (type == 2) { modelName = "face_landmarker_pts5.csta"; }
	seeta::v6::FaceLandmarker faceLandmarker(ModelSetting(modelPath + modelName, SEETA_DEVICE_AUTO));

	*size = faceLandmarker.number();
	std::vector<SeetaPointF> _points = faceLandmarker.mark(img, faceRect);

	*size = _points.size();
	if (!_points.empty()) {
		SeetaPointF* points = (SeetaPointF*)malloc((*size) * sizeof(SeetaPointF));
		SeetaPointF* start = points;
		for (auto iter = _points.begin(); iter != _points.end(); iter++)
		{
			*start = *iter;
			start++;
		}
		return points;
	}
	return 0;
}

// 提取人脸特征值
View_Api std::vector<float>* Extract(SeetaImageData& img, SeetaPointF* points, int* size, int type = 0)
{
	string modelName = "face_recognizer.csta";
	if (type == 1) { modelName = "face_recognizer_mask.csta"; }
	if (type == 2) { modelName = "face_recognizer_light.csta"; }
	seeta::v6::FaceRecognizer faceRecognizer(ModelSetting(modelPath + modelName));

	*size = faceRecognizer.GetExtractFeatureSize();
	std::shared_ptr<float> _features(new float[*size], std::default_delete<float[]>());
	faceRecognizer.Extract(img, points, _features.get());

	float* features = _features.get();
	std::vector<float> results{ features, features + *size };
	return &results;
}

// 人脸特征值相似度计算
View_Api float CalculateSimilarity(float* leftFeatures, float* rightFeatures, int type = 0)
{
	string modelName = "face_recognizer.csta";
	if (type == 1) { modelName = "face_recognizer_mask.csta"; }
	if (type == 2) { modelName = "face_recognizer_light.csta"; }
	seeta::v6::FaceRecognizer faceRecognizer(ModelSetting(modelPath + modelName, SEETA_DEVICE_CPU, 0));
	float similarity = faceRecognizer.CalculateSimilarity(leftFeatures, rightFeatures);
	return similarity;
}

/***************************************************************************************************************/
// 活体检测器
// 活体检测 - 单帧
View_Api int AntiSpoofing(SeetaImageData& img, SeetaRect faceRect, SeetaPointF* points, bool global)
{
	ModelSetting setting;
	setting.append(modelPath + "fas_first.csta");
	//启用全局检测能力
	if (global) { setting.append(modelPath + "fas_second.csta"); }
	seeta::v6::FaceAntiSpoofing faceAntiSpoofing(setting);
	auto state = faceAntiSpoofing.Predict(img, faceRect, points);
	return state;
}

static seeta::v6::FaceAntiSpoofing* faceAntiSpoofing = nullptr;
// 活体检测 - 视频
View_Api int AntiSpoofingVideo(SeetaImageData& img, SeetaRect faceRect, SeetaPointF* points, bool global)
{
	if (faceAntiSpoofing == nullptr) {
		ModelSetting setting;
		setting.append(modelPath + "fas_first.csta");
		// 启用全局检测能力
		if (global) { setting.append(modelPath + "fas_second.csta"); }
		faceAntiSpoofing = new FaceAntiSpoofing(setting);
	}

	auto status = faceAntiSpoofing->PredictVideo(img, faceRect, points);
	if (status != FaceAntiSpoofing::Status::DETECTING)
	{
		delete faceAntiSpoofing;
		faceAntiSpoofing = nullptr;
	}
	return status;
}

// 获取跟踪的人脸个数
View_Api SeetaTrackingFaceInfo* FaceTrack(SeetaImageData& img, int* size, bool stable = false, int interval = 10, int faceSize = 20, float threshold = 0.9, int type = 0)
{
	seeta::v6::FaceTracker faceTracker(ModelSetting(modelPath + (type == 0 ? "face_detector.csta" : "mask_detector.csta")), img.width, img.height);
	faceTracker.SetVideoStable(stable);
	faceTracker.SetMinFaceSize(faceSize);
	faceTracker.SetThreshold(threshold);
	faceTracker.SetInterval(interval);

	auto cfaces = faceTracker.Track(img);
	std::vector<SeetaTrackingFaceInfo> faces(cfaces.data, cfaces.data + cfaces.size);
	*size = faces.size();
	SeetaTrackingFaceInfo* _infos = (SeetaTrackingFaceInfo*)malloc((*size) * sizeof(SeetaTrackingFaceInfo));
	int index = 0;
	for (auto& face : faces)
	{
		_infos[index] = face;
		index++;
	}
	return _infos;
}


// 亮度评估
View_Api void Quality_Brightness(SeetaImageData& img, SeetaRect faceRect, SeetaPointF* points, int pointsLength, int* level, float* score, float v0 = 70, float v1 = 100, float v2 = 210, float v3 = 230)
{
	seeta::v3::QualityOfBrightness quality_Brightness(v0, v1, v2, v3);
	auto result = quality_Brightness.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 清晰度评估
View_Api void Quality_Clarity(SeetaImageData& img, SeetaRect faceRect, SeetaPointF* points, int pointsLength, int* level, float* score, float low = 0.1f, float high = 0.2f)
{
	seeta::v3::QualityOfClarity quality_Clarity(low, high);
	auto result = quality_Clarity.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 完整度评估
View_Api void Quality_Integrity(SeetaImageData& img, SeetaRect faceRect, SeetaPointF* points, int pointsLength, int* level, float* score, float low = 10, float high = 1.5f)
{
	seeta::v3::QualityOfIntegrity quality_Integrity(low, high);
	auto result = quality_Integrity.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 姿态评估
View_Api void Quality_Pose(SeetaImageData& img, SeetaRect faceRect, SeetaPointF* points, int pointsLength, int* level, float* score)
{
	auto quality_Pose = new seeta::v3::QualityOfPose();
	auto result = quality_Pose->check(img, faceRect, points, pointsLength);
	delete quality_Pose;

	*level = result.level;
	*score = result.score;
}

// 姿态 (深度)评估
View_Api void Quality_PoseEx(SeetaImageData& img, SeetaRect faceRect, SeetaPointF* points, int pointsLength, int* level, float* score,
	float yawLow = 25, float yawHigh = 10, float pitchLow = 20, float pitchHigh = 10, float rollLow = 33.33f, float rollHigh = 16.67f)
{
	seeta::v3::QualityOfPoseEx quality_PoseEx(ModelSetting(modelPath + "pose_estimation.csta"));
	quality_PoseEx.set(QualityOfPoseEx::YAW_LOW_THRESHOLD, yawLow);
	quality_PoseEx.set(QualityOfPoseEx::YAW_HIGH_THRESHOLD, yawHigh);
	quality_PoseEx.set(QualityOfPoseEx::PITCH_LOW_THRESHOLD, pitchLow);
	quality_PoseEx.set(QualityOfPoseEx::PITCH_HIGH_THRESHOLD, pitchHigh);
	quality_PoseEx.set(QualityOfPoseEx::ROLL_LOW_THRESHOLD, rollLow);
	quality_PoseEx.set(QualityOfPoseEx::ROLL_HIGH_THRESHOLD, rollHigh);

	auto result = quality_PoseEx.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 分辨率评估
View_Api void Quality_Resolution(SeetaImageData& img, SeetaRect faceRect, SeetaPointF* points, int pointsLength, int* level, float* score, float low = 80, float high = 120)
{
	seeta::v3::QualityOfResolution quality_Resolution(low, high);
	auto result = quality_Resolution.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 清晰度 (深度)评估
View_Api void Quality_ClarityEx(SeetaImageData& img, SeetaRect faceRect, SeetaPointF* points, int pointsLength, int* level, float* score, float blur_thresh = 0.8f)
{
	seeta::QualityOfClarityEx quality_ClarityEx(blur_thresh, modelPath);
	auto result = quality_ClarityEx.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 遮挡评估
View_Api void Quality_NoMask(SeetaImageData& img, SeetaRect faceRect, SeetaPointF* points, int pointsLength, int* level, float* score)
{
	seeta::QualityOfNoMask quality_NoMask(modelPath);
	auto result = quality_NoMask.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

/******人脸属性***********************************************************************************************/
// 年龄预测
View_Api int PredictAge(SeetaImageData& img, SeetaPointF* points, int pointsLength)
{
	seeta::v6::AgePredictor age_Predictor(ModelSetting(modelPath + "age_predictor.csta"));
	int age = 0;
	bool result = age_Predictor.PredictAgeWithCrop(img, points, age);

	if (result) { return age; }
	else { return -1; }
}

// 年龄预测
View_Api int PredictGender(SeetaImageData& img, SeetaPointF* points, int pointsLength)
{
	seeta::v6::GenderPredictor gender_Predictor(ModelSetting(modelPath + "gender_predictor.csta"));
	GenderPredictor::GENDER gender = GenderPredictor::GENDER::MALE;
	auto result = gender_Predictor.PredictGenderWithCrop(img, points, gender);

	if (result) { return gender; }
	else { return -1; }
}

// 年龄预测
View_Api void EyeDetector(SeetaImageData& img, SeetaPointF* points, int pointsLength, EyeStateDetector::EYE_STATE& left_eye, EyeStateDetector::EYE_STATE& right_eye)
{
	seeta::v6::EyeStateDetector eyeState_Detector(ModelSetting(modelPath + "eye_state.csta"));
	eyeState_Detector.Detect(img, points, left_eye, right_eye);
}