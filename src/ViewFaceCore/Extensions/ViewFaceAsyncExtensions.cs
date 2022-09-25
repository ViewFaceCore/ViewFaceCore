using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using ViewFaceCore.Models;

namespace ViewFaceCore.Core
{
    /// <summary>
    /// API异步扩展
    /// <para>参考: <a href="https://docs.microsoft.com/zh-cn/dotnet/standard/async-in-depth#deeper-dive-into-task-and-taskt-for-a-cpu-bound-operation">深入了解绑定 CPU 的操作的任务和 Task&lt;T&gt;</a></para>
    /// </summary>
    public static class ViewFaceAsyncExtensions
    {
        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回人脸的信息。
        /// </summary>
        /// <param name="viewFace"><see cref="FaceDetector"/> 对象</param>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。</returns>
        public static Task<FaceInfo[]> DetectAsync(this FaceDetector viewFace, FaceImage image)
            => Task.Run(() => viewFace.Detect(image));

        /// <summary>
        /// 识别 <paramref name="image"/> 中指定的人脸信息 <paramref name="info"/> 的关键点坐标。
        /// </summary>
        /// <param name="viewFace"><see cref="FaceLandmarker"/> 对象</param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">指定的人脸信息</param>
        /// <returns>若失败，则返回结果 Length == 0</returns>
        public static Task<FaceMarkPoint[]> MarkAsync(this FaceLandmarker viewFace, FaceImage image, FaceInfo info)
            => Task.Run(() => viewFace.Mark(image, info));

        /// <summary>
        /// 提取人脸特征值。
        /// </summary>
        /// <param name="viewFace"><see cref="FaceRecognizer"/> 对象</param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">人脸关键点数据</param>
        /// <returns></returns>
        public static Task<float[]> ExtractAsync(this FaceRecognizer viewFace, FaceImage image, FaceMarkPoint[] points)
            => Task.Run(() => viewFace.Extract(image, points));

        /// <summary>
        /// 活体检测器。(单帧图片)
        /// </summary>
        /// <param name="viewFace"><see cref="FaceAntiSpoofing"/> 对象</param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector.Detect(FaceImage)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceLandmarker.Mark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>活体检测状态</returns>
        public static Task<AntiSpoofingResult> PredictAsync(this FaceAntiSpoofing viewFace, FaceImage image, FaceInfo info, FaceMarkPoint[] points)
            => Task.Run(() => viewFace.Predict(image, info, points));

        /// <summary>
        /// 活体检测器。(视频帧图片)
        /// </summary>
        /// <param name="viewFace"><see cref="FaceAntiSpoofing"/> 对象</param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector.Detect(FaceImage)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceLandmarker.Mark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>如果为 <see cref="AntiSpoofingStatus.Detecting"/>，则说明需要继续调用此方法，传入更多的图片</returns>
        public static Task<AntiSpoofingResult> PredictVideoAsync(this FaceAntiSpoofing viewFace, FaceImage image, FaceInfo info, FaceMarkPoint[] points)
            => Task.Run(() => viewFace.PredictVideo(image, info, points));

        /// <summary>
        /// 人脸质量评估
        /// </summary>
        /// <param name="viewFace"><see cref="FaceQuality"/> 对象</param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector.Detect(FaceImage)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceLandmarker.Mark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <param name="type">质量评估类型</param>
        /// <returns></returns>
        public static Task<QualityResult> DetectAsync(this FaceQuality viewFace, FaceImage image, FaceInfo info, FaceMarkPoint[] points, QualityType type)
            => Task.Run(() => viewFace.Detect(image, info, points, type));

        /// <summary>
        /// 年龄预测（自动裁剪）
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.age_predictor">age_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="viewFace"><see cref="AgePredictor"/> 对象</param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="FaceLandmarker.Mark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>-1: 预测失败失败，其它: 预测的年龄。</returns>
        public static Task<int> PredictAgeWithCropAsync(this AgePredictor viewFace, FaceImage image, FaceMarkPoint[] points)
            => Task.Run(() => viewFace.PredictAgeWithCrop(image, points));

        /// <summary>
        /// 年龄预测
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.age_predictor">age_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="viewFace"><see cref="AgePredictor"/> 对象</param>
        /// <param name="image">人脸图像信息</param>
        /// <returns>-1: 预测失败失败，其它: 预测的年龄。</returns>
        public static Task<int> PredictAgeAsync(this AgePredictor viewFace, FaceImage image)
            => Task.Run(() => viewFace.PredictAge(image));

        /// <summary>
        /// 性别预测（自动裁剪）
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.gender_predictor">gender_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="viewFace"><see cref="GenderPredictor"/> 对象</param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="FaceLandmarker.Mark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>性别。<see cref="Gender.Unknown"/> 代表识别失败</returns>
        public static Task<Gender> PredictGenderWithCropAsync(this GenderPredictor viewFace, FaceImage image, FaceMarkPoint[] points)
            => Task.Run(() => viewFace.PredictGenderWithCrop(image, points));

        /// <summary>
        /// 性别预测
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.gender_predictor">gender_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="viewFace"><see cref="GenderPredictor"/> 对象</param>
        /// <param name="image">人脸图像信息</param>
        /// <returns>性别。<see cref="Gender.Unknown"/> 代表识别失败</returns>
        public static Task<Gender> PredictGenderAsync(this GenderPredictor viewFace, FaceImage image)
            => Task.Run(() => viewFace.PredictGender(image));

        /// <summary>
        /// 眼睛状态检测。
        /// <para>
        /// 眼睛的左右是相对图片内容而言的左右。<br />
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.eye_state">eye_state.csta</a>
        /// </para>
        /// </summary>
        /// <param name="viewFace"><see cref="EyeStateDetector"/> 对象</param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="FaceLandmarker.Mark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns></returns>
        public static Task<EyeStateResult> DetectAsync(this EyeStateDetector viewFace, FaceImage image, FaceMarkPoint[] points)
            => Task.Run(() => viewFace.Detect(image, points));

        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回可跟踪的人脸信息。
        /// </summary>
        /// <param name="viewFace"><see cref="FaceTracker"/> 对象</param>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。</returns>
        public static Task<FaceTrackInfo[]> TrackAsync(this FaceTracker viewFace, FaceImage image)
            => Task.Run(() => viewFace.Track(image));

        /// <summary>
        /// 戴口罩人脸识别
        /// </summary>
        /// <param name="viewFace"><see cref="FaceTracker"/> 对象</param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector.Detect(FaceImage)"/> 获取</para></param>
        /// <returns></returns>
        public static Task<PlotMaskResult> PlotMaskAsync(this MaskDetector viewFace, FaceImage image, FaceInfo info)
            => Task.Run(() => viewFace.Detect(image, info));
    }
}
