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
#include "seeta/MaskDetector.h"

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

//释放标识，避免被释放后二次创建对象，然后就忘了释放
static bool _isDispose = false;

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
View_Api void Free(shared_ptr<void>* address)
{
	free(address);
}

static seeta::v6::FaceDetector* _faceDetector = nullptr;
// 获取人脸数量
View_Api SeetaFaceInfo* Detector(const SeetaImageData& img, int* size, const double faceSize = 20, const double threshold = 0.9, const double maxWidth = 2000, const double maxHeight = 2000)
{
	if (_isDispose) {
		throw "The current object has been disposed.";
	}
	if (_faceDetector == nullptr) {
		_faceDetector = new seeta::v6::FaceDetector(ModelSetting(modelPath + "face_detector.csta", SEETA_DEVICE_AUTO));
		_faceDetector->set(FaceDetector::Property::PROPERTY_MIN_FACE_SIZE, faceSize);
		_faceDetector->set(FaceDetector::Property::PROPERTY_THRESHOLD, threshold);
		_faceDetector->set(FaceDetector::Property::PROPERTY_MAX_IMAGE_WIDTH, maxWidth);
		_faceDetector->set(FaceDetector::Property::PROPERTY_MAX_IMAGE_HEIGHT, maxHeight);
	}
	else {
		double epsilon = 0.000001;
		if (fabs(faceSize - _faceDetector->get(FaceDetector::Property::PROPERTY_MIN_FACE_SIZE)) >= epsilon) {
			_faceDetector->set(FaceDetector::Property::PROPERTY_MIN_FACE_SIZE, faceSize);
		}
		if (fabs(threshold - _faceDetector->get(FaceDetector::Property::PROPERTY_THRESHOLD)) >= epsilon) {
			_faceDetector->set(FaceDetector::Property::PROPERTY_THRESHOLD, threshold);
		}
		if (fabs(maxWidth - _faceDetector->get(FaceDetector::Property::PROPERTY_MAX_IMAGE_WIDTH)) >= epsilon) {
			_faceDetector->set(FaceDetector::Property::PROPERTY_MAX_IMAGE_WIDTH, maxWidth);
		}
		if (fabs(maxHeight - _faceDetector->get(FaceDetector::Property::PROPERTY_MAX_IMAGE_HEIGHT)) >= epsilon) {
			_faceDetector->set(FaceDetector::Property::PROPERTY_MAX_IMAGE_HEIGHT, maxHeight);
		}
	}
	auto faces = _faceDetector->detect_v2(img);
	*size = faces.size();
	SeetaFaceInfo* _infos = (SeetaFaceInfo*)malloc((*size) * sizeof(SeetaFaceInfo));
	if (_infos == nullptr) {
		return 0;
	}
	for (int i = 0; i < faces.size(); i++) {
		_infos[i] = faces[i];
	}
	return _infos;
}

static seeta::v6::FaceLandmarker* faceLandmarker_type_0 = nullptr;
static seeta::v6::FaceLandmarker* faceLandmarker_type_1 = nullptr;
static seeta::v6::FaceLandmarker* faceLandmarker_type_2 = nullptr;

// 人脸关键点器
View_Api SeetaPointF* FaceMark(const SeetaImageData& img, const SeetaRect faceRect, long* size, const int type = 0)
{
	if (_isDispose) {
		throw "The current object has been disposed.";
	}
	seeta::v6::FaceLandmarker* faceLandmarker = nullptr;
	switch (type)
	{
	case 0:
		if (faceLandmarker_type_0 == nullptr) {
			faceLandmarker_type_0 = new seeta::v6::FaceLandmarker(ModelSetting(modelPath + "face_landmarker_pts68.csta", SEETA_DEVICE_AUTO));
		}
		faceLandmarker = faceLandmarker_type_0;
		break;
	case 1:
		if (faceLandmarker_type_1 == nullptr) {
			faceLandmarker_type_1 = new seeta::v6::FaceLandmarker(ModelSetting(modelPath + "face_landmarker_mask_pts5.csta", SEETA_DEVICE_AUTO));
		}
		faceLandmarker = faceLandmarker_type_1;
		break;
	case 2:
		if (faceLandmarker_type_2 == nullptr) {
			faceLandmarker_type_2 = new seeta::v6::FaceLandmarker(ModelSetting(modelPath + "face_landmarker_pts5.csta", SEETA_DEVICE_AUTO));
		}
		faceLandmarker = faceLandmarker_type_2;
		break;
	default:
		throw "Unsupport type.";
	}

	*size = faceLandmarker->number();
	std::vector<SeetaPointF> _points = faceLandmarker->mark(img, faceRect);

	*size = _points.size();
	if (!_points.empty()) {
		SeetaPointF* points = (SeetaPointF*)malloc((*size) * sizeof(SeetaPointF));
		if (points == nullptr) {
			return 0;
		}
		SeetaPointF* start = points;
		for (auto iter = _points.begin(); iter != _points.end(); iter++) {
			*start = *iter;
			start++;
		}
		return points;
	}
	return 0;
}

static seeta::v6::FaceRecognizer* _faceRecognizer_type_0 = nullptr;
static seeta::v6::FaceRecognizer* _faceRecognizer_type_1 = nullptr;
static seeta::v6::FaceRecognizer* _faceRecognizer_type_2 = nullptr;

// 提取人脸特征值
View_Api float* Extract(const SeetaImageData& img, const SeetaPointF* points, int* size, const int type = 0)
{
	if (_isDispose) {
		throw "The current object has been disposed.";
	}
	seeta::v6::FaceRecognizer* faceRecognizer = nullptr;
	switch (type)
	{
	case 0:
		if (_faceRecognizer_type_0 == nullptr) {
			_faceRecognizer_type_0 = new seeta::v6::FaceRecognizer(ModelSetting(modelPath + "face_recognizer.csta", SEETA_DEVICE_AUTO));
		}
		faceRecognizer = _faceRecognizer_type_0;
		break;
	case 1:
		if (_faceRecognizer_type_1 == nullptr) {
			_faceRecognizer_type_1 = new seeta::v6::FaceRecognizer(ModelSetting(modelPath + "face_recognizer_mask.csta", SEETA_DEVICE_AUTO));
		}
		faceRecognizer = _faceRecognizer_type_1;
		break;
	case 2:
		if (_faceRecognizer_type_2 == nullptr) {
			_faceRecognizer_type_2 = new seeta::v6::FaceRecognizer(ModelSetting(modelPath + "face_recognizer_light.csta", SEETA_DEVICE_AUTO));
		}
		faceRecognizer = _faceRecognizer_type_2;
		break;
	default:
		throw "Unsupport type.";
	}
	*size = faceRecognizer->GetExtractFeatureSize();
	std::shared_ptr<float> _features(new float[*size], std::default_delete<float[]>());
	faceRecognizer->Extract(img, points, _features.get());

	float* source = _features.get();
	float* features = (float*)malloc(*size * sizeof(float));
	if (features != nullptr)
	{
		memcpy(features, source, *size * sizeof(float));
	}
	return features;
}

// 人脸特征值相似度计算
View_Api float Compare(const float* lhs, const float* rhs, int size) {
	float sum = 0;
	for (int i = 0; i < size; ++i) {
		sum += *lhs * *rhs;
		++lhs;
		++rhs;
	}
	return sum;
}

/***************************************************************************************************************/

static seeta::v6::FaceAntiSpoofing* _faceAntiSpoofing = nullptr;
static seeta::v6::FaceAntiSpoofing* _faceAntiSpoofing_global = nullptr;
// 活体检测器
// 活体检测 - 单帧
View_Api int AntiSpoofing(const SeetaImageData& img, const SeetaRect faceRect, const SeetaPointF* points, const bool global)
{
	seeta::v6::FaceAntiSpoofing* faceAntiSpoofing = nullptr;
	//启用全局检测能力
	if (global) {
		if (_faceAntiSpoofing_global == nullptr) {
			ModelSetting setting;
			setting.append(modelPath + "fas_first.csta");
			setting.append(modelPath + "fas_second.csta");
			_faceAntiSpoofing_global = new seeta::v6::FaceAntiSpoofing(setting);
		}
		faceAntiSpoofing = _faceAntiSpoofing_global;
	}
	else {
		if (_faceAntiSpoofing == nullptr) {
			ModelSetting setting;
			setting.append(modelPath + "fas_first.csta");
			_faceAntiSpoofing = new seeta::v6::FaceAntiSpoofing(setting);
		}
		faceAntiSpoofing = _faceAntiSpoofing;
	}
	auto state = faceAntiSpoofing->Predict(img, faceRect, points);
	return state;
}

static seeta::v6::FaceAntiSpoofing* _faceAntiSpoofing_video = nullptr;
static seeta::v6::FaceAntiSpoofing* _faceAntiSpoofing_video_global = nullptr;
// 活体检测 - 视频
View_Api int AntiSpoofingVideo(const SeetaImageData& img, const SeetaRect faceRect, const SeetaPointF* points, const bool global)
{
	seeta::v6::FaceAntiSpoofing* faceAntiSpoofing = nullptr;
	//启用全局检测能力
	if (global) {
		if (_faceAntiSpoofing_video_global == nullptr) {
			ModelSetting setting;
			setting.append(modelPath + "fas_first.csta");
			setting.append(modelPath + "fas_second.csta");
			_faceAntiSpoofing_video_global = new seeta::v6::FaceAntiSpoofing(setting);
		}
		faceAntiSpoofing = _faceAntiSpoofing_video_global;
	}
	else {
		if (_faceAntiSpoofing_video == nullptr) {
			ModelSetting setting;
			setting.append(modelPath + "fas_first.csta");
			_faceAntiSpoofing_video = new seeta::v6::FaceAntiSpoofing(setting);
		}
		faceAntiSpoofing = _faceAntiSpoofing_video;
	}

	auto status = faceAntiSpoofing->PredictVideo(img, faceRect, points);
	if (status != FaceAntiSpoofing::Status::DETECTING)
	{
		faceAntiSpoofing->ResetVideo();
	}
	return status;
}

// 获取人脸追踪句柄
View_Api seeta::v6::FaceTracker* GetFaceTrackHandler(const int width
	, const int height
	, const int type = 0
	, const bool stable = false
	, const int interval = 10
	, const int faceSize = 20
	, const float threshold = 0.9) {
	seeta::v6::FaceTracker* faceTracker = new seeta::v6::FaceTracker(ModelSetting(modelPath + (type == 0 ? "face_detector.csta" : "mask_detector.csta")), width, height);
	faceTracker->SetVideoStable(stable);
	faceTracker->SetMinFaceSize(faceSize);
	faceTracker->SetThreshold(threshold);
	faceTracker->SetInterval(interval);
	return faceTracker;
}

// 获取跟踪的人脸个数
View_Api SeetaTrackingFaceInfo* FaceTrack(const seeta::v6::FaceTracker* faceTracker, const SeetaImageData& img, int* size)
{
	if (faceTracker == nullptr) {
		return 0;
	}
	auto cfaces = faceTracker->Track(img);
	std::vector<SeetaTrackingFaceInfo> faces(cfaces.data, cfaces.data + cfaces.size);
	*size = faces.size();
	SeetaTrackingFaceInfo* _infos = new SeetaTrackingFaceInfo[*size];
	for (int i = 0; i < faces.size(); i++)
	{
		_infos[i] = faces[i];
	}
	return _infos;
}

//重置追踪结果
View_Api void FaceTrackReset(seeta::v6::FaceTracker* faceTracker) {
	if (faceTracker == nullptr) {
		return;
	}
	faceTracker->Reset();
}

// 释放追踪对象
View_Api void FaceTrackDispose(seeta::v6::FaceTracker* faceTracker) {
	if (faceTracker == nullptr) {
		return;
	}
	delete faceTracker;
}


// 亮度评估
View_Api void Quality_Brightness(const SeetaImageData& img
	, const SeetaRect faceRect
	, const SeetaPointF* points
	, const int pointsLength
	, int* level
	, float* score
	, const float v0 = 70
	, const float v1 = 100
	, const float v2 = 210
	, const float v3 = 230)
{
	seeta::v3::QualityOfBrightness quality_Brightness(v0, v1, v2, v3);
	auto result = quality_Brightness.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 清晰度评估
View_Api void Quality_Clarity(const SeetaImageData& img, const SeetaRect faceRect, const  SeetaPointF* points, const int pointsLength, int* level, float* score, const float low = 0.1f, const float high = 0.2f)
{
	seeta::v3::QualityOfClarity quality_Clarity(low, high);
	auto result = quality_Clarity.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 完整度评估
View_Api void Quality_Integrity(const SeetaImageData& img, const SeetaRect faceRect, const SeetaPointF* points, const int pointsLength, int* level, float* score, const float low = 10, const float high = 1.5f)
{
	seeta::v3::QualityOfIntegrity quality_Integrity(low, high);
	auto result = quality_Integrity.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 姿态评估
View_Api void Quality_Pose(const SeetaImageData& img, const SeetaRect faceRect, const SeetaPointF* points, const int pointsLength, int* level, float* score)
{
	seeta::v3::QualityOfPose quality_Pose;
	auto result = quality_Pose.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 姿态 (深度)评估
View_Api void Quality_PoseEx(const SeetaImageData& img, const SeetaRect faceRect, const SeetaPointF* points, const int pointsLength, int* level, float* score,
	const float yawLow = 25, const float yawHigh = 10, const float pitchLow = 20, const float pitchHigh = 10, const float rollLow = 33.33f, const float rollHigh = 16.67f)
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
View_Api void Quality_Resolution(const SeetaImageData& img, const SeetaRect faceRect, const SeetaPointF* points, const int pointsLength, int* level, float* score, const float low = 80, const float high = 120)
{
	seeta::v3::QualityOfResolution quality_Resolution(low, high);
	auto result = quality_Resolution.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 清晰度 (深度)评估
View_Api void Quality_ClarityEx(const SeetaImageData& img, const SeetaRect faceRect, const SeetaPointF* points, const int pointsLength, int* level, float* score, const float blur_thresh = 0.8f)
{
	seeta::QualityOfClarityEx quality_ClarityEx(blur_thresh, modelPath);
	auto result = quality_ClarityEx.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 遮挡评估
View_Api void Quality_NoMask(const SeetaImageData& img, const SeetaRect faceRect, const SeetaPointF* points, const int pointsLength, int* level, float* score)
{
	seeta::QualityOfNoMask quality_NoMask(modelPath);
	auto result = quality_NoMask.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

/******人脸属性***********************************************************************************************/
// 年龄预测
View_Api int PredictAge(const SeetaImageData& img, const SeetaPointF* points, const int pointsLength)
{
	seeta::v6::AgePredictor age_Predictor(ModelSetting(modelPath + "age_predictor.csta"));
	int age = 0;
	bool result = age_Predictor.PredictAgeWithCrop(img, points, age);

	if (result) { return age; }
	else { return -1; }
}

// 年龄预测
View_Api int PredictGender(const SeetaImageData& img, const SeetaPointF* points, const int pointsLength)
{
	seeta::v6::GenderPredictor gender_Predictor(ModelSetting(modelPath + "gender_predictor.csta"));
	GenderPredictor::GENDER gender = GenderPredictor::GENDER::MALE;
	auto result = gender_Predictor.PredictGenderWithCrop(img, points, gender);

	if (result) { return gender; }
	else { return -1; }
}

// 眼睛状态检测
View_Api void EyeDetector(const SeetaImageData& img, const SeetaPointF* points, const int pointsLength, EyeStateDetector::EYE_STATE& left_eye, EyeStateDetector::EYE_STATE& right_eye)
{
	seeta::v6::EyeStateDetector eyeState_Detector(ModelSetting(modelPath + "eye_state.csta"));
	eyeState_Detector.Detect(img, points, left_eye, right_eye);
}

template<typename T>
void _dispose(T& ptr) {
	if (ptr != nullptr) {
		try {
			delete ptr;
			ptr = nullptr;
		}
		catch (int e) {

		}
	}
}

View_Api void Dispose() {
	_isDispose = true;
	//获取人脸数量
	_dispose(_faceDetector);
	//人脸关键点
	_dispose(faceLandmarker_type_0);
	_dispose(faceLandmarker_type_1);
	_dispose(faceLandmarker_type_2);
	//提取人脸特征值
	_dispose(_faceRecognizer_type_0);
	_dispose(_faceRecognizer_type_1);
	_dispose(_faceRecognizer_type_2);
	//活体检测
	_dispose(_faceAntiSpoofing);
	_dispose(_faceAntiSpoofing_global);
	_dispose(_faceAntiSpoofing_video);
	_dispose(_faceAntiSpoofing_video_global);

}

