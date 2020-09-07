# ViewFaceCore API 文档

## ViewFace 类
 命名空间 : ViewFaceCore.Sharp  
 人脸识别类

```
public class ViewFace
```

### 构造函数

|名称|说明|
|-|-|
| `ViewFace()` | 使用默认的模型目录初始化人脸识别类 |
| `ViewFace(string)` | 使用指定的模型目录初始化人脸识别类 |
| `ViewFace(LogCallBack)` | 使用指定的日志回调函数初始化人脸识别类 |
| `ViewFace(string ,LogCallBack)` | 使用指定的模型目录、日志回调函数初始化人脸识别类 |


### 属性

| 名称 | 类型 | 说明 | 默认值 |
|-|-|-|-|
| `DetectorConfig` | `FaceDetectorConfig` | 获取或设置人脸检测器配置 | `new FaceDetectorConfig()` |
| `ModelPath` | `string` | 获取或设置模型路径 | `./model/` |
| `FaceType` | `FaceType` | 获取或设置人脸类型 | `FaceType.Normal` |
| `MarkType` | `MarkType` | 获取或设置人脸关键点类型 | `MarkType.Light` |
| `TrackerConfig` | `FaceTrackerConfig` | 获取或设置人脸跟踪器的配置 | `new FaceTrackerConfig()` |
| `QualityConfig` | `QualityConfig` | 获取或设置质量评估器的配置 | `new QualityConfig()` |

### 方法

|签名|说明|
|-|-|
| `FaceInfo[] FaceDetector(Bitmap)` | 识别 `Bitmap` 中的人脸，并返回人脸的信息 |
| `Task<FaceInfo[]> FaceDetectorAsync(Bitmap)` | `FaceDetector` 的异步版本 |
|  |  |
| `FaceMarkPoint[] FaceMark(Bitmap, FaceInfo)` | 识别 `Bitmap` 中指定的人脸信息 FaceInfo 的关键点坐标 |
| `Task<FaceMarkPoint[]> FaceMarkAsync(Bitmap, FaceInfo)` | `FaceMark` 的异步版本 |
|  |  |
| `float[] Extract(Bitmap, FaceMarkPoint[])` | 提取人脸特征值 |
| `Task<float[]> ExtractAsync(Bitmap, FaceMarkPoint[])` | `Extract` 的异步版本 |
|  |  |
| `float Similarity(float[], float[])` | 计算特征值相似度 |
| `Task<float> SimilarityAsync(float[], float[])` | `Similarity` 的异步版本 |
|  |  |
| `bool IsSelf(float)` | 判断相似度是否为同一个人 |
| `AntiSpoofingStatus AntiSpoofing(Bitmap, FaceInfo, FaceMarkPoint[], bool)` | 活体检测器 - 单帧 |
| `Task<AntiSpoofingStatus> AntiSpoofingAsync(Bitmap, FaceInfo, FaceMarkPoint[], bool)` | `AntiSpoofing` 的异步版本 |
|  |  |
| `bool IsSelf(float)` | 判断相似度是否为同一个人 |
| `AntiSpoofingStatus AntiSpoofingVideo(Bitmap, FaceInfo, FaceMarkPoint[], bool)` | 活体检测器 - 视频帧 |
| `Task<AntiSpoofingStatus> AntiSpoofingVideoAsync(Bitmap, FaceInfo, FaceMarkPoint[], bool)` | `AntiSpoofingVideo` 的异步版本 |
|  |  |
| `FaceTrackInfo[] FaceTrack(Bitmap)` | 识别 `Bitmap` 中的人脸，并返回可跟踪的人脸信息 |
| `Task<FaceTrackInfo[]> FaceTrackAsync(Bitmap)` | `FaceTrack` 的异步版本 |
|  |  |
| `QualityResult FaceQuality(Bitmap, FaceInfo, FaceMarkPoint[], QualityType)` | 人脸质量评估 |
| `Task<QualityResult> FaceQualityAsync(Bitmap, FaceInfo, FaceMarkPoint[], QualityType)` | `FaceQuality` 的异步版本 |
|  |  |
| `int FaceAgePredictor(Bitmap, FaceMarkPoint[])` | 年龄预测 |
| `Task<int> FaceAgePredictorAsync(Bitmap, FaceMarkPoint[])` | `FaceAgePredictor` 的异步版本 |
|  |  |
| `Gender FaceGenderPredictor(Bitmap, FaceMarkPoint[])` | 性别预测 |
| `Task<Gender> FaceGenderPredictorAsync(Bitmap, FaceMarkPoint[])` | `FaceGenderPredictor` 的异步版本 |
|  |  |
| `EyeStateResult FaceEyeStateDetector(Bitmap, FaceMarkPoint[])` | 眼睛状态检测 |
| `Task<EyeStateResult> FaceEyeStateDetectorAsync(Bitmap, FaceMarkPoint[])` | `FaceEyeStateDetector` 的异步版本 |
|  |  |


### 可能用到的类型

*建设中...*