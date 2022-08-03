#include "bridge.h"

#if WINDOWS

#define STDCALL _stdcall
#define View_Api extern "C" __declspec(dllexport)

#elif LINUX

#define STDCALL __attribute__((stdcall))
#define View_Api extern "C"

#endif // WINDOWS or LINUX

using namespace std;
using namespace seeta;

#pragma region Common

/// <summary>
/// 释放
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="ptr"></param>
template <typename T>
void _dispose(T &ptr)
{
	if (ptr != nullptr)
	{
		try
		{
			delete ptr;
			ptr = nullptr;
		}
		catch (int e)
		{
		}
	}
}

// 模型所在路径
string modelPath = "./viewfacecore/models/";

// 设置人脸模型目录
View_Api void SetModelPath(const char *path)
{
	modelPath = path;
}

// 获取人脸模型目录
View_Api void GetModelPath(char **path)
{
	strcpy(*path, modelPath.c_str());
}

// 释放由 malloc 分配的内存
View_Api void Free(void *address)
{
	try
	{
		free(address);
	}
	catch (int e)
	{
	}
}

#pragma endregion

#pragma region FaceDetector

/// <summary>
/// 创建人脸识别句柄
/// </summary>
/// <param name="faceSize"></param>
/// <param name="threshold"></param>
/// <param name="maxWidth"></param>
/// <param name="maxHeight"></param>
/// <returns></returns>
View_Api seeta::v6::FaceDetector *GetFaceDetectorHandler(const double faceSize = 20, const double threshold = 0.9, const double maxWidth = 2000, const double maxHeight = 2000, const SeetaDevice deviceType = SEETA_DEVICE_AUTO)
{
	seeta::v6::FaceDetector *faceDetector = new seeta::v6::FaceDetector(ModelSetting(modelPath + "face_detector.csta", deviceType));
	faceDetector->set(FaceDetector::Property::PROPERTY_MIN_FACE_SIZE, faceSize);
	faceDetector->set(FaceDetector::Property::PROPERTY_THRESHOLD, threshold);
	faceDetector->set(FaceDetector::Property::PROPERTY_MAX_IMAGE_WIDTH, maxWidth);
	faceDetector->set(FaceDetector::Property::PROPERTY_MAX_IMAGE_HEIGHT, maxHeight);
	return faceDetector;
}

/// <summary>
/// 获取人脸数量
/// </summary>
/// <param name="faceDetector"></param>
/// <param name="img"></param>
/// <param name="size"></param>
/// <returns></returns>
View_Api SeetaFaceInfo *FaceDetector(seeta::v6::FaceDetector *handler, const SeetaImageData &img, int *size)
{
	if (handler == nullptr)
	{
		return 0;
	}
	auto faces = handler->detect_v2(img);
	*size = faces.size();
	SeetaFaceInfo *_infos = (SeetaFaceInfo *)malloc((*size) * sizeof(SeetaFaceInfo));
	if (_infos == nullptr)
	{
		return 0;
	}
	for (int i = 0; i < faces.size(); i++)
	{
		_infos[i] = faces[i];
	}
	return _infos;
}

/// <summary>
/// 释放人脸识别句柄
/// </summary>
/// <param name="faceDetector"></param>
/// <returns></returns>
View_Api void DisposeFaceDetector(seeta::v6::FaceDetector *handler)
{
	_dispose(handler);
}

#pragma endregion

#pragma region MaskDetector

/// <summary>
/// 创建人脸识别句柄（口罩识别）
/// </summary>
/// <param name="deviceType"></param>
/// <returns></returns>
View_Api seeta::v2::MaskDetector *GetMaskDetectorHandler(const SeetaDevice deviceType = SEETA_DEVICE_AUTO)
{
	return new seeta::v2::MaskDetector(ModelSetting(modelPath + "mask_detector.csta", deviceType));
}

/// <summary>
/// 口罩检测
/// </summary>
/// <param name="handler"></param>
/// <param name="img"></param>
/// <param name="size"></param>
/// <returns></returns>
View_Api bool PlotMask(seeta::v2::MaskDetector *handler, const SeetaImageData &img, const SeetaRect faceRect, float *score)
{
	if (handler == nullptr)
	{
		return 0;
	}
	bool result = handler->detect(img, faceRect, score);
	return result;
}

/// <summary>
/// 释放口罩识别句柄
/// </summary>
/// <param name="faceDetector"></param>
/// <returns></returns>
View_Api void DisposeMaskDetector(seeta::v2::MaskDetector *handler)
{
	_dispose(handler);
}

#pragma endregion

#pragma region FaceMark

/// <summary>
/// 获取人脸关键点句柄
/// </summary>
/// <param name="type"></param>
/// <returns></returns>
View_Api seeta::v6::FaceLandmarker *GetFaceLandmarkerHandler(const int type = 0, const SeetaDevice deviceType = SEETA_DEVICE_AUTO)
{
	switch (type)
	{
	case 0:
		return new seeta::v6::FaceLandmarker(ModelSetting(modelPath + "face_landmarker_pts68.csta", deviceType));
	case 1:
		return new seeta::v6::FaceLandmarker(ModelSetting(modelPath + "face_landmarker_mask_pts5.csta", deviceType));
	case 2:
		return new seeta::v6::FaceLandmarker(ModelSetting(modelPath + "face_landmarker_pts5.csta", deviceType));
	default:
		throw "Unsupport type.";
	}
}

/// <summary>
/// 人脸关键点器
/// </summary>
/// <param name="img"></param>
/// <param name="faceRect"></param>
/// <param name="size"></param>
/// <param name="type"></param>
/// <returns></returns>
View_Api SeetaPointF *FaceMark(seeta::v6::FaceLandmarker *handler, const SeetaImageData &img, const SeetaRect faceRect, long *size)
{
	if (handler == nullptr)
	{
		return 0;
	}
	*size = handler->number();
	std::vector<SeetaPointF> _points = handler->mark(img, faceRect);

	*size = _points.size();
	if (!_points.empty())
	{
		SeetaPointF *points = (SeetaPointF *)malloc((*size) * sizeof(SeetaPointF));
		if (points == nullptr)
		{
			return 0;
		}
		SeetaPointF *start = points;
		for (auto iter = _points.begin(); iter != _points.end(); iter++)
		{
			*start = *iter;
			start++;
		}
		return points;
	}
	return 0;
}

View_Api void DisposeFaceLandmarker(seeta::v6::FaceLandmarker *handler)
{
	_dispose(handler);
}

#pragma endregion

#pragma region FaceRecognizer

/// <summary>
/// 获取人脸特征值句柄
/// </summary>
/// <param name="type"></param>
/// <returns></returns>
View_Api seeta::v6::FaceRecognizer *GetFaceRecognizerHandler(const int type = 0, const SeetaDevice deviceType = SEETA_DEVICE_AUTO)
{
	switch (type)
	{
	case 0:
		return new seeta::v6::FaceRecognizer(ModelSetting(modelPath + "face_recognizer.csta", deviceType));
	case 1:
		return new seeta::v6::FaceRecognizer(ModelSetting(modelPath + "face_recognizer_mask.csta", deviceType));
	case 2:
		return new seeta::v6::FaceRecognizer(ModelSetting(modelPath + "face_recognizer_light.csta", deviceType));
	default:
		throw "Unsupport type.";
	}
}

/// <summary>
/// 提取人脸特征值
/// </summary>
/// <param name="img"></param>
/// <param name="points"></param>
/// <param name="size"></param>
/// <param name="type"></param>
/// <returns></returns>
View_Api float *FaceRecognizerExtract(seeta::v6::FaceRecognizer *handler, const SeetaImageData &img, const SeetaPointF *points, int *size)
{
	if (handler == nullptr)
	{
		return 0;
	}
	*size = handler->GetExtractFeatureSize();
	std::shared_ptr<float> _features(new float[*size], std::default_delete<float[]>());
	handler->Extract(img, points, _features.get());

	float *source = _features.get();
	float *features = (float *)malloc(*size * sizeof(float));
	if (features != nullptr)
	{
		memcpy(features, source, *size * sizeof(float));
	}
	return features;
}

View_Api void DisposeFaceRecognizer(seeta::v6::FaceRecognizer *handler)
{
	_dispose(handler);
}

/// <summary>
/// 人脸特征值相似度计算
/// </summary>
/// <param name="lhs"></param>
/// <param name="rhs"></param>
/// <param name="size"></param>
/// <returns></returns>
View_Api float Compare(const float *lhs, const float *rhs, int size)
{
	float sum = 0;
	for (int i = 0; i < size; ++i)
	{
		sum += *lhs * *rhs;
		++lhs;
		++rhs;
	}
	return sum;
}

#pragma endregion

#pragma region FaceAntiSpoofing

View_Api seeta::v6::FaceAntiSpoofing *GetFaceAntiSpoofingHandler(const int videoFrameCount = 10
	, const float boxThresh = 0.8
	, const float clarity = 0.3
	, const float reality = 0.8
	, const bool global = false
	, const SeetaDevice deviceType = SEETA_DEVICE_AUTO)
{
	ModelSetting setting(modelPath + "fas_first.csta", deviceType);
	if (global)
	{
		setting.append(modelPath + "fas_second.csta");
	}
	seeta::v6::FaceAntiSpoofing *faceAntiSpoofing = new seeta::v6::FaceAntiSpoofing(setting);
	faceAntiSpoofing->SetVideoFrameCount(videoFrameCount);
	faceAntiSpoofing->SetBoxThresh(boxThresh);
	faceAntiSpoofing->SetThreshold(clarity, reality);
	return faceAntiSpoofing;
}

/// <summary>
/// 活体检测 - 单帧
/// </summary>
/// <param name="img"></param>
/// <param name="faceRect"></param>
/// <param name="points"></param>
/// <param name="global"></param>
/// <returns></returns>
View_Api int AntiSpoofing(seeta::v6::FaceAntiSpoofing *handler
	, const SeetaImageData &img
	, const SeetaRect faceRect
	, const SeetaPointF *points
	, float* clarity
	, float* reality)
{
	if (handler == nullptr)
	{
		return -1;
	}
	FaceAntiSpoofing::Status state = handler->Predict(img, faceRect, points);
	handler->GetPreFrameScore(clarity, reality);
	return state;
}

/// <summary>
/// 活体检测 - 视频
/// </summary>
/// <param name="img"></param>
/// <param name="faceRect"></param>
/// <param name="points"></param>
/// <param name="global"></param>
/// <returns></returns>
View_Api int AntiSpoofingVideo(seeta::v6::FaceAntiSpoofing *handler
	, const SeetaImageData &img
	, const SeetaRect faceRect
	, const SeetaPointF *points
	, float* clarity
	, float* reality)
{
	if (handler == nullptr)
	{
		return -1;
	}
	auto status = handler->PredictVideo(img, faceRect, points);
	handler->GetPreFrameScore(clarity, reality);
	if (status != FaceAntiSpoofing::Status::DETECTING)
	{
		handler->ResetVideo();
	}
	return status;
}

View_Api void DisposeFaceAntiSpoofing(seeta::v6::FaceAntiSpoofing *handler)
{
	_dispose(handler);
}

#pragma endregion

#pragma region FaceTracker

/// <summary>
/// 获取人脸追踪句柄
/// </summary>
/// <param name="width"></param>
/// <param name="height"></param>
/// <param name="type"></param>
/// <param name="stable"></param>
/// <param name="interval"></param>
/// <param name="faceSize"></param>
/// <param name="threshold"></param>
/// <returns></returns>
View_Api seeta::v6::FaceTracker *GetFaceTrackerHandler(const int width, const int height, const bool stable = false, const int interval = 10, const int faceSize = 20, const float threshold = 0.9, const SeetaDevice deviceType = SEETA_DEVICE_AUTO)
{
	seeta::v6::FaceTracker *faceTracker = new seeta::v6::FaceTracker(ModelSetting(modelPath + "face_detector.csta", deviceType), width, height);
	faceTracker->SetVideoStable(stable);
	faceTracker->SetMinFaceSize(faceSize);
	faceTracker->SetThreshold(threshold);
	faceTracker->SetInterval(interval);
	return faceTracker;
}

/// <summary>
/// 获取跟踪的人脸个数
/// </summary>
/// <param name="faceTracker"></param>
/// <param name="img"></param>
/// <param name="size"></param>
/// <returns></returns>
View_Api SeetaTrackingFaceInfo *FaceTrack(seeta::v6::FaceTracker *handler, const SeetaImageData &img, int *size)
{
	if (handler == nullptr)
	{
		return 0;
	}
	auto cfaces = handler->Track(img);
	std::vector<SeetaTrackingFaceInfo> faces(cfaces.data, cfaces.data + cfaces.size);
	*size = faces.size();
	SeetaTrackingFaceInfo *_infos = new SeetaTrackingFaceInfo[*size];
	for (int i = 0; i < faces.size(); i++)
	{
		_infos[i] = faces[i];
	}
	return _infos;
}

/// <summary>
/// 重置追踪
/// </summary>
/// <param name="faceTracker"></param>
/// <returns></returns>
View_Api void FaceTrackReset(seeta::v6::FaceTracker *handler)
{
	if (handler == nullptr)
	{
		return;
	}
	handler->Reset();
}

/// <summary>
/// 释放追踪对象
/// </summary>
/// <param name="faceTracker"></param>
/// <returns></returns>
View_Api void DisposeFaceTracker(seeta::v6::FaceTracker *handler)
{
	_dispose(handler);
}

#pragma endregion

#pragma region Quality

// 亮度评估
View_Api void Quality_Brightness(const SeetaImageData &img, const SeetaRect faceRect, const SeetaPointF *points, const int pointsLength, int *level, float *score, const float v0 = 70, const float v1 = 100, const float v2 = 210, const float v3 = 230)
{
	seeta::v3::QualityOfBrightness quality_Brightness(v0, v1, v2, v3);
	auto result = quality_Brightness.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 清晰度评估
View_Api void Quality_Clarity(const SeetaImageData &img, const SeetaRect faceRect, const SeetaPointF *points, const int pointsLength, int *level, float *score, const float low = 0.1f, const float high = 0.2f)
{
	seeta::v3::QualityOfClarity quality_Clarity(low, high);
	auto result = quality_Clarity.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 完整度评估
View_Api void Quality_Integrity(const SeetaImageData &img, const SeetaRect faceRect, const SeetaPointF *points, const int pointsLength, int *level, float *score, const float low = 10, const float high = 1.5f)
{
	seeta::v3::QualityOfIntegrity quality_Integrity(low, high);
	auto result = quality_Integrity.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 姿态评估
View_Api void Quality_Pose(const SeetaImageData &img, const SeetaRect faceRect, const SeetaPointF *points, const int pointsLength, int *level, float *score)
{
	seeta::v3::QualityOfPose quality_Pose;
	auto result = quality_Pose.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 姿态 (深度)评估
View_Api void Quality_PoseEx(const SeetaImageData &img, const SeetaRect faceRect, const SeetaPointF *points, const int pointsLength, int *level, float *score,
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
View_Api void Quality_Resolution(const SeetaImageData &img, const SeetaRect faceRect, const SeetaPointF *points, const int pointsLength, int *level, float *score, const float low = 80, const float high = 120)
{
	seeta::v3::QualityOfResolution quality_Resolution(low, high);
	auto result = quality_Resolution.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 清晰度 (深度)评估
View_Api void Quality_ClarityEx(const SeetaImageData &img, const SeetaRect faceRect, const SeetaPointF *points, const int pointsLength, int *level, float *score, const float blur_thresh = 0.8f)
{
	seeta::QualityOfClarityEx quality_ClarityEx(blur_thresh, modelPath);
	auto result = quality_ClarityEx.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

// 遮挡评估
View_Api void Quality_NoMask(const SeetaImageData &img, const SeetaRect faceRect, const SeetaPointF *points, const int pointsLength, int *level, float *score)
{
	seeta::QualityOfNoMask quality_NoMask(modelPath);
	auto result = quality_NoMask.check(img, faceRect, points, pointsLength);

	*level = result.level;
	*score = result.score;
}

#pragma endregion

#pragma region AgePredictor / GenderPredictor / EyeStateDetector

#pragma region AgePredictor

/// <summary>
/// 获取年龄预测句柄
/// </summary>
/// <returns></returns>
View_Api seeta::v6::AgePredictor *GetAgePredictorHandler(const SeetaDevice deviceType = SEETA_DEVICE_AUTO)
{
	return new seeta::v6::AgePredictor(ModelSetting(modelPath + "age_predictor.csta", deviceType));
}

/// <summary>
/// 年龄预测
/// </summary>
/// <param name="img"></param>
/// <param name="points"></param>
/// <returns></returns>
View_Api int PredictAge(seeta::v6::AgePredictor *handler, const SeetaImageData &img, const SeetaPointF *points)
{
	if (handler == nullptr)
	{
		return -1;
	}
	int age = 0;
	bool result = handler->PredictAgeWithCrop(img, points, age);
	if (result)
	{
		return age;
	}
	else
	{
		return -1;
	}
}

/// <summary>
/// 释放年龄预测句柄
/// </summary>
/// <param name="handler"></param>
/// <returns></returns>
View_Api void DisposeAgePredictor(seeta::v6::AgePredictor *handler)
{
	_dispose(handler);
}

#pragma endregion

#pragma region GenderPredictor

/// <summary>
/// 获取性别预测句柄
/// </summary>
/// <returns></returns>
View_Api seeta::v6::GenderPredictor *GetGenderPredictorHandler(const SeetaDevice deviceType = SEETA_DEVICE_AUTO)
{
	return new seeta::v6::GenderPredictor(ModelSetting(modelPath + "gender_predictor.csta", deviceType));
}

/// <summary>
/// 性别预测
/// </summary>
/// <param name="img"></param>
/// <param name="points"></param>
/// <returns></returns>
View_Api int PredictGender(seeta::v6::GenderPredictor *handler, const SeetaImageData &img, const SeetaPointF *points)
{
	if (handler == nullptr)
	{
		return -1;
	}
	GenderPredictor::GENDER gender = GenderPredictor::GENDER::MALE;
	auto result = handler->PredictGenderWithCrop(img, points, gender);
	if (result)
	{
		return gender;
	}
	else
	{
		return -1;
	}
}

/// <summary>
/// 释放性别预测句柄
/// </summary>
/// <param name="handler"></param>
/// <returns></returns>
View_Api void DisposeGenderPredictor(seeta::v6::GenderPredictor *handler)
{
	_dispose(handler);
}

#pragma endregion

#pragma region EyeStateDetector

/// <summary>
/// 获取眼睛状态检测句柄
/// </summary>
/// <returns></returns>
View_Api seeta::v6::EyeStateDetector *GetEyeStateDetectorHandler(const SeetaDevice deviceType = SEETA_DEVICE_AUTO)
{
	return new seeta::v6::EyeStateDetector(ModelSetting(modelPath + "eye_state.csta", deviceType));
}

/// <summary>
/// 眼睛状态检测
/// </summary>
/// <param name="img"></param>
/// <param name="points"></param>
/// <returns></returns>
View_Api void EyeStateDetector(seeta::v6::EyeStateDetector *handler, const SeetaImageData &img, const SeetaPointF *points, EyeStateDetector::EYE_STATE &left_eye, EyeStateDetector::EYE_STATE &right_eye)
{
	if (handler == nullptr)
	{
		return;
	}
	handler->Detect(img, points, left_eye, right_eye);
}

/// <summary>
/// 释放眼睛状态检测句柄
/// </summary>
/// <param name="handler"></param>
/// <returns></returns>
View_Api void DisposeEyeStateDetector(seeta::v6::EyeStateDetector *handler)
{
	_dispose(handler);
}

#pragma endregion

#pragma endregion
