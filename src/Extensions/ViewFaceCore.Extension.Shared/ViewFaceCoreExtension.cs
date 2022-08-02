using System;
using ViewFaceCore.Core;
using ViewFaceCore.Model;


namespace ViewFaceCore
{
    public static class ViewFaceCoreExtension
    {
        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回人脸的信息。
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="DetectorConfig"/> 重新检测。</returns>
        public static FaceInfo[] Detect<T>(this FaceDetector viewFace, T image) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.Detect(faceImage);
            }
        }

        /// <summary>
        /// 识别 <paramref name="image"/> 中指定的人脸信息 <paramref name="info"/> 的关键点坐标。
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">指定的人脸信息</param>
        /// <exception cref="MarkException"/>
        /// <returns>若失败，则返回结果 Length == 0</returns>
        public static FaceMarkPoint[] Mark<T>(this FaceLandmarker viewFace, T image, FaceInfo info) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.Mark(faceImage, info);
            }
        }

        /// <summary>
        /// 提取人脸特征值。
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">人脸关键点数据</param>
        /// <returns></returns>
        public static float[] Extract<T>(this FaceRecognizer viewFace, T image, FaceMarkPoint[] points) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.Extract(faceImage, points);
            }
        }

        /// <summary>
        /// 活体检测器。(单帧图片)
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector(FaceImage)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>活体检测状态</returns>
        public static AntiSpoofingStatus AntiSpoofing<T>(this FaceAntiSpoofing viewFace, T image, FaceInfo info, FaceMarkPoint[] points) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.AntiSpoofing(faceImage, info, points);
            }
        }

        /// <summary>
        /// 活体检测器。(视频帧图片)
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector(FaceImage)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>如果为 <see cref="AntiSpoofingStatus.Detecting"/>，则说明需要继续调用此方法，传入更多的图片</returns>
        public static AntiSpoofingStatus AntiSpoofingVideo<T>(this FaceAntiSpoofing viewFace, T image, FaceInfo info, FaceMarkPoint[] points) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.AntiSpoofingVideo(faceImage, info, points);
            }
        }

        /// <summary>
        /// 人脸质量评估
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector(FaceImage)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <param name="type">质量评估类型</param>
        /// <returns></returns>
        public static QualityResult Detect<T>(this FaceQuality viewFace, T image, FaceInfo info, FaceMarkPoint[] points, QualityType type) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.Detect(faceImage, info, points, type);
            }
        }

        /// <summary>
        /// 年龄预测。
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.age_predictor">age_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>-1: 预测失败失败，其它: 预测的年龄。</returns>
        public static int PredictAge<T>(this AgePredictor viewFace, T image, FaceMarkPoint[] points) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.PredictAge(faceImage, points);
            }
        }

        /// <summary>
        /// 性别预测。
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.gender_predictor">gender_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>性别。<see cref="Gender.Unknown"/> 代表识别失败</returns>
        public static Gender PredictGender<T>(this GenderPredictor viewFace, T image, FaceMarkPoint[] points) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.PredictGender(faceImage, points);
            }
        }

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
        public static EyeStateResult Detect<T>(this EyeStateDetector viewFace, T image, FaceMarkPoint[] points) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.Detect(faceImage, points);
            }
        }

        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回可跟踪的人脸信息。
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="TrackerConfig"/> 重新检测。</returns>
        public static FaceTrackInfo[] Track<T>(this FaceTracker viewFace, T image) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.Track(faceImage);
            }
        }

        /// <summary>
        /// 戴口罩人脸识别
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="maskDetector"></param>
        /// <param name="image"></param>
        /// <param name="info"></param>
        /// <param name="score"></param>
        /// <returns></returns>
        public static bool PlotMask<T>(this MaskDetector maskDetector, T image, FaceInfo info, out float score) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return maskDetector.PlotMask(faceImage, info, out score);
            }
        }
    }
}
