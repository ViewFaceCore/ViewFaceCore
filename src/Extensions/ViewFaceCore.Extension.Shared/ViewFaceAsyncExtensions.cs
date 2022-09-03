using System;
using System.Threading;
using System.Threading.Tasks;
using ViewFaceCore.Exceptions;
using ViewFaceCore.Model;

namespace ViewFaceCore.Core
{
    /// <summary>
    /// 异步扩展，对于 CPU 绑定的操作直接使用 <see cref="Task.Run(Action)"/> 进行包装。
    /// <para>参考: <a href="https://docs.microsoft.com/zh-cn/dotnet/standard/async-in-depth#deeper-dive-into-task-and-taskt-for-a-cpu-bound-operation">深入了解绑定 CPU 的操作的任务和 Task&lt;T&gt;</a></para>
    /// </summary>
    public static class ViewFaceAsyncExtensions
    {
        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回人脸的信息。
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="DetectorConfig"/> 重新检测。</returns>
        public static Task<FaceInfo[]> DetectAsync<T>(this FaceDetector viewFace, T image) where T : class
            => Run(() => viewFace.Detect(image));

        /// <summary>
        /// 识别 <paramref name="image"/> 中指定的人脸信息 <paramref name="info"/> 的关键点坐标。
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">指定的人脸信息</param>
        /// <exception cref="MarkException"/>
        /// <returns>若失败，则返回结果 Length == 0</returns>
        public static Task<FaceMarkPoint[]> MarkAsync<T>(this FaceLandmarker viewFace, T image, FaceInfo info) where T : class
            => Run(() => viewFace.Mark(image, info));

        /// <summary>
        /// 提取人脸特征值。
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">人脸关键点数据</param>
        /// <returns></returns>
        public static Task<float[]> ExtractAsync<T>(this FaceRecognizer viewFace, T image, FaceMarkPoint[] points) where T : class
            => Run(() => viewFace.Extract(image, points));

        /// <summary>
        /// 活体检测器。(单帧图片)
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector(FaceImage)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>活体检测状态</returns>
        public static Task<AntiSpoofingResult> AntiSpoofingAsync<T>(this FaceAntiSpoofing viewFace, T image, FaceInfo info, FaceMarkPoint[] points) where T : class
            => Run(() => viewFace.AntiSpoofing(image, info, points));

        /// <summary>
        /// 活体检测器。(视频帧图片)
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector(FaceImage)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>如果为 <see cref="AntiSpoofingStatus.Detecting"/>，则说明需要继续调用此方法，传入更多的图片</returns>
        public static Task<AntiSpoofingResult> AntiSpoofingVideoAsync<T>(this FaceAntiSpoofing viewFace, T image, FaceInfo info, FaceMarkPoint[] points) where T : class
            => Run(() => viewFace.AntiSpoofingVideo(image, info, points));

        /// <summary>
        /// 人脸质量评估
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector(FaceImage)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <param name="type">质量评估类型</param>
        /// <returns></returns>
        public static Task<QualityResult> DetectAsync<T>(this FaceQuality viewFace, T image, FaceInfo info, FaceMarkPoint[] points, QualityType type) where T : class
            => Run(() => viewFace.Detect(image, info, points, type));

        /// <summary>
        /// 年龄预测。
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.age_predictor">age_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>-1: 预测失败失败，其它: 预测的年龄。</returns>
        public static Task<int> PredictAgeAsync<T>(this AgePredictor viewFace, T image, FaceMarkPoint[] points) where T : class
            => Run(() => viewFace.PredictAge(image, points));

        /// <summary>
        /// 性别预测。
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.gender_predictor">gender_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>性别。<see cref="Gender.Unknown"/> 代表识别失败</returns>
        public static Task<Gender> PredictGenderAsync<T>(this GenderPredictor viewFace, T image, FaceMarkPoint[] points) where T : class
            => Run(() => viewFace.PredictGender(image, points));

        /// <summary>
        /// 眼睛状态检测。
        /// <para>
        /// 眼睛的左右是相对图片内容而言的左右。<br />
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.eye_state">eye_state.csta</a>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns></returns>
        public static Task<EyeStateResult> DetectAsync<T>(this EyeStateDetector viewFace, T image, FaceMarkPoint[] points) where T : class
            => Run(() => viewFace.Detect(image, points));

        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回可跟踪的人脸信息。
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="TrackerConfig"/> 重新检测。</returns>
        public static Task<FaceTrackInfo[]> TrackAsync<T>(this FaceTracker viewFace, T image) where T : class
            => Run(() => viewFace.Track(image));

        /// <summary>
        /// 戴口罩人脸识别
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="viewFace"></param>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info"></param>
        /// <returns></returns>
        public static Task<PlotMaskResult> PlotMaskAsync<T>(this MaskDetector viewFace, T image, FaceInfo info) where T : class
            => Run(() => viewFace.PlotMask(image, info));

        /// <summary>
        /// 封装Task.Run
        /// </summary>
        /// <typeparam name="TResult"></typeparam>
        /// <param name="function"></param>
        /// <returns></returns>
        private static Task<TResult> Run<TResult>(Func<TResult> function)
        {
#if NET45_OR_GREATER || NETCOREAPP || NETSTANDARD
            return Task.Factory.StartNew<TResult>(function, CancellationToken.None, TaskCreationOptions.DenyChildAttach, TaskScheduler.Default);
#else
            return Task.Factory.StartNew<TResult>(function, CancellationToken.None, TaskCreationOptions.None, TaskScheduler.Default);
#endif
        }
    }
}