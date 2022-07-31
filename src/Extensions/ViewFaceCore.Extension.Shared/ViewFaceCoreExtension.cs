using System;
using System.Collections.Generic;
using System.Text;
using ViewFaceCore.Model;


namespace ViewFaceCore
{
    public static class ViewFaceCoreExtension
    {
        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回人脸的信息。
        /// <para>
        /// 可以通过 <see cref="DetectorConfig"/> 属性对人脸检测器进行配置，以应对不同场景的图片。
        /// </para>
        /// <para>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> <see langword="||"/> <see cref="FaceType.Light"/></c> 时， 需要模型： <a href="https://www.nuget.org/packages/ViewFaceCore.model.face_detector/">face_detector.csta</a><br/>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/></c> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.mask_detector">mask_detector.csta</a><br/>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="DetectorConfig"/> 重新检测。</returns>
        public static IEnumerable<FaceInfo> FaceDetector<T>(this ViewFace viewFace, T image) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                foreach (var info in viewFace.FaceDetector(faceImage))
                {
                    yield return info;
                }
            }
        }

        /// <summary>
        /// 识别 <paramref name="image"/> 中指定的人脸信息 <paramref name="info"/> 的关键点坐标。
        /// <para>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_pts68">face_landmarker_pts68.csta</a><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_mask_pts5">face_landmarker_mask_pts5.csta</a><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Light"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_pts5">face_landmarker_pts5.csta</a><br/>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">指定的人脸信息</param>
        /// <exception cref="MarkException"/>
        /// <returns>若失败，则返回结果 Length == 0</returns>
        public static IEnumerable<FaceMarkPoint> FaceMark<T>(this ViewFace viewFace, T image, FaceInfo info) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                foreach (var point in viewFace.FaceMark(faceImage, info))
                {
                    yield return point;
                }
            }
        }

        /// <summary>
        /// 提取人脸特征值。
        /// <para>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_recognizer">face_recognizer.csta</a><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_recognizer_mask">face_recognizer_mask.csta</a><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Light"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_recognizer_light">face_recognizer_light.csta</a><br/>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">人脸关键点数据</param>
        /// <returns></returns>
        public static float[] Extract<T>(this ViewFace viewFace, T image, IEnumerable<FaceMarkPoint> points) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.Extract(faceImage, points);
            }
        }

        /// <summary>
        /// 活体检测器。(单帧图片)
        /// <para>
        /// 当 <paramref name="global"/> <see langword="= false"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.fas_first">fas_first.csta</a><br/>
        /// 当 <paramref name="global"/> <see langword="= true"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.fas_second">fas_second.csta</a>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector(FaceImage)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <param name="global">是否启用全局检测能力</param>
        /// <returns>活体检测状态</returns>
        public static AntiSpoofingStatus AntiSpoofing<T>(this ViewFace viewFace, T image, FaceInfo info, IEnumerable<FaceMarkPoint> points, bool global = false) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.AntiSpoofing(faceImage, info, points, global);
            }
        }

        /// <summary>
        /// 活体检测器。(视频帧图片)
        /// <para>
        /// 当 <paramref name="global"/> <see langword="= false"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.fas_first">fas_first.csta</a><br/>
        /// 当 <paramref name="global"/> <see langword="= true"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.fas_second">fas_second.csta</a>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector(FaceImage)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <param name="global">是否启用全局检测能力</param>
        /// <returns>如果为 <see cref="AntiSpoofingStatus.Detecting"/>，则说明需要继续调用此方法，传入更多的图片</returns>
        public static AntiSpoofingStatus AntiSpoofingVideo<T>(this ViewFace viewFace, T image, FaceInfo info, IEnumerable<FaceMarkPoint> points, bool global = false) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.AntiSpoofingVideo(faceImage, info, points, global);
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
        public static QualityResult FaceQuality<T>(this ViewFace viewFace, T image, FaceInfo info, IEnumerable<FaceMarkPoint> points, QualityType type) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.FaceQuality(faceImage, info, points, type);
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
        public static int FaceAgePredictor<T>(this ViewFace viewFace, T image, IEnumerable<FaceMarkPoint> points) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.FaceAgePredictor(faceImage, points);
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
        public static Gender FaceGenderPredictor<T>(this ViewFace viewFace, T image, IEnumerable<FaceMarkPoint> points) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.FaceGenderPredictor(faceImage, points);
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
        public static EyeStateResult FaceEyeStateDetector<T>(this ViewFace viewFace, T image, IEnumerable<FaceMarkPoint> points) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return viewFace.FaceEyeStateDetector(faceImage, points);
            }
        }

        #region 人脸追踪

        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回可跟踪的人脸信息。
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="TrackerConfig"/> 重新检测。</returns>
        public static FaceTrackInfo[] Track<T>(this FaceTrack faceTrack, T image) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                return faceTrack.Track(faceImage);
            }
        }

        #endregion

    }
}
