#if NET45_OR_GREATER

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using ViewFaceCore.Extension;
using ViewFaceCore.Plus;
using ViewFaceCore.Sharp.Configs;
using ViewFaceCore.Sharp.Exceptions;
using ViewFaceCore.Sharp.Model;

namespace ViewFaceCore.Sharp
{
    public partial class ViewFace
    {
        /// <summary>
        /// 识别 <paramref name="bitmap"/> 中的人脸，并返回人脸的信息。
        /// <para><see cref="FaceDetector(Bitmap)"/> 的异步版本。</para>
        /// <para>
        /// 可以通过 <see cref="DetectorConfig"/> 属性对人脸检测器进行配置，以应对不同场景的图片。
        /// </para>
        /// <para>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> <see langword="||"/> <see cref="FaceType.Light"/></c> 时， 需要模型：<see langword="face_detector.csta"/><br/>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/></c> 时， 需要模型：<see langword="mask_detector.csta"/><br/>
        /// </para>
        /// </summary>
        /// <param name="bitmap">包含人脸的图片</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="DetectorConfig"/> 重新检测。</returns>
        public async Task<FaceInfo[]> FaceDetectorAsync(Bitmap bitmap)
            => await Task.Run(() => FaceDetector(bitmap));

        /// <summary>
        /// 识别 <paramref name="bitmap"/> 中指定的人脸信息 <paramref name="info"/> 的关键点坐标。
        /// <para><see cref="FaceMark(Bitmap, FaceInfo)"/> 的异步版本。</para>
        /// <para>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> 时， 需要模型：<see langword="face_landmarker_pts68.csta"/><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/> 时， 需要模型：<see langword="face_landmarker_mask_pts5.csta"/><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Light"/> 时， 需要模型：<see langword="face_landmarker_pts5.csta"/><br/>
        /// </para>
        /// </summary>
        /// <param name="bitmap">包含人脸的图片</param>
        /// <param name="info">指定的人脸信息</param>
        /// <exception cref="MarkException"/>
        /// <returns>若失败，则返回结果 Length == 0</returns>
        public async Task<FaceMarkPoint[]> FaceMarkAsync(Bitmap bitmap, FaceInfo info)
            => await Task.Run(() => FaceMark(bitmap, info));

        /// <summary>
        /// 提取人脸特征值。
        /// <para><see cref="Extract(Bitmap, FaceMarkPoint[])"/> 的异步版本。</para>
        /// <para>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> 时， 需要模型：<see langword="face_recognizer.csta"/><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/> 时， 需要模型：<see langword="face_recognizer_mask.csta"/><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Light"/> 时， 需要模型：<see langword="face_recognizer_light.csta"/><br/>
        /// </para>
        /// </summary>
        /// <param name="bitmap"></param>
        /// <param name="points"></param>
        /// <exception cref="ExtractException"/>
        /// <returns></returns>
        public async Task<float[]> ExtractAsync(Bitmap bitmap, FaceMarkPoint[] points)
            => await Task.Run(() => Extract(bitmap, points));

        /// <summary>
        /// 计算特征值相似度。
        /// <para><see cref="Similarity(float[], float[])"/> 的异步版本。</para>
        /// <para>只能计算相同 <see cref="FaceType"/> 计算出的特征值</para>
        /// <para>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> 时， 需要模型：<see langword="face_recognizer.csta"/><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/> 时， 需要模型：<see langword="face_recognizer_mask.csta"/><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Light"/> 时， 需要模型：<see langword="face_recognizer_light.csta"/><br/>
        /// </para>
        /// </summary>
        /// <exception cref="ArgumentException"/>
        /// <exception cref="ArgumentNullException"/>
        /// <param name="leftFeatures"></param>
        /// <param name="rightFeatures"></param>
        /// <returns></returns>
        public async Task<float> SimilarityAsync(float[] leftFeatures, float[] rightFeatures)
            => await Task.Run(() => Similarity(leftFeatures, rightFeatures));

        /// <summary>
        /// 活体检测器。
        /// <para><see cref="AntiSpoofing(Bitmap, FaceInfo, FaceMarkPoint[], bool)"/> 的异步版本。</para>
        /// <para>
        /// 单帧图片，由 <paramref name="global"/> 指定是否启用全局检测能力 <br />
        /// 需通过 <see cref="FaceDetector(Bitmap)"/> 获取 <paramref name="info"/> 参数<br/>
        /// 通过 <see cref="FaceMark(Bitmap, FaceInfo)"/> 获取与 <paramref name="info"/> 参数对应的 <paramref name="points"/>
        /// </para>
        /// <para>
        /// 当 <paramref name="global"/> <see langword="= false"/> 时， 需要模型：<see langword="fas_first.csta"/><br/>
        /// 当 <paramref name="global"/> <see langword="= true"/> 时， 需要模型：<see langword="fas_second.csta"/>
        /// </para>
        /// </summary>
        /// <param name="bitmap"></param>
        /// <param name="info"></param>
        /// <param name="points"></param>
        /// <param name="global"></param>
        /// <returns></returns>
        public async Task<AntiSpoofingStatus> AntiSpoofingAsync(Bitmap bitmap, FaceInfo info, FaceMarkPoint[] points, bool global = false)
            => await Task.Run(() => AntiSpoofing(bitmap, info, points, global));

        /// <summary>
        /// 活体检测器。
        /// <para><see cref="AntiSpoofingVideo(Bitmap, FaceInfo, FaceMarkPoint[], bool)"/> 的异步版本。</para>
        /// <para>
        /// 视频帧图片，由 <paramref name="global"/> 指定是否启用全局检测能力 <br />
        /// 需通过 <see cref="FaceDetector(Bitmap)"/> 获取 <paramref name="info"/> 参数<br/>
        /// 通过 <see cref="FaceMark(Bitmap, FaceInfo)"/> 获取与 <paramref name="info"/> 参数对应的 <paramref name="points"/>
        /// </para>
        /// <para>
        /// 当 <paramref name="global"/> <see langword="= false"/> 时， 需要模型：<see langword="fas_first.csta"/><br/>
        /// 当 <paramref name="global"/> <see langword="= true"/> 时， 需要模型：<see langword="fas_second.csta"/>
        /// </para>
        /// <para>如果返回结果为 <see cref="AntiSpoofingStatus.Detecting"/>，则说明需要继续调用此方法，传入更多的图片</para>
        /// </summary>
        /// <param name="bitmap"></param>
        /// <param name="info"></param>
        /// <param name="points"></param>
        /// <param name="global">是否启用全局检测能力</param>
        /// <returns></returns>
        public async Task<AntiSpoofingStatus> AntiSpoofingVideoAsync(Bitmap bitmap, FaceInfo info, FaceMarkPoint[] points, bool global)
            => await Task.Run(() => AntiSpoofingVideo(bitmap, info, points, global));

        /// <summary>
        /// 识别 <paramref name="bitmap"/> 中的人脸，并返回可跟踪的人脸信息。
        /// <para><see cref="FaceTrack(Bitmap)"/> 的异步版本。</para>
        /// <para>
        /// 可以通过 <see cref="TrackerConfig"/> 属性对人脸检测器进行配置，以应对不同场景的图片。
        /// </para>
        /// <para>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> <see langword="||"/> <see cref="FaceType.Light"/></c> 时， 需要模型：<see langword="face_detector.csta"/><br/>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/></c> 时， 需要模型：<see langword="mask_detector.csta"/><br/>
        /// </para>
        /// </summary>
        /// <param name="bitmap">包含人脸的图片</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="TrackerConfig"/> 重新检测。</returns>
        public async Task<FaceTrackInfo[]> FaceTrackAsync(Bitmap bitmap)
            => await Task.Run(() => FaceTrack(bitmap));

        /// <summary>
        /// 人脸质量评估
        /// <para><see cref="FaceQuality(Bitmap, FaceInfo, FaceMarkPoint[], QualityType)"/> 的异步版本。</para>
        /// </summary>
        /// <param name="bitmap"></param>
        /// <param name="info"></param>
        /// <param name="points"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        public async Task<QualityResult> FaceQualityAsync(Bitmap bitmap, FaceInfo info, FaceMarkPoint[] points, QualityType type)
            => await Task.Run(() => FaceQuality(bitmap, info, points, type));

        /// <summary>
        /// 年龄预测。
        /// <para>
        /// <see cref="FaceAgePredictor(Bitmap, FaceMarkPoint[])"/> 的异步版本。<br />
        /// 需要模型 <see langword="age_predictor.csta"/> 
        /// </para>
        /// </summary>
        /// <param name="bitmap">待识别的图像</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <returns></returns>
        public async Task<int> FaceAgePredictorAsync(Bitmap bitmap, FaceMarkPoint[] points)
            => await Task.Run(() => FaceAgePredictor(bitmap, points));

        /// <summary>
        /// 性别预测。
        /// <para>
        /// <see cref="FaceGenderPredictor(Bitmap, FaceMarkPoint[])"/> 的异步版本。<br />
        /// 需要模型 <see langword="gender_predictor.csta"/> 
        /// </para>
        /// </summary>
        /// <param name="bitmap">待识别的图像</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <returns></returns>
        public async Task<Gender> FaceGenderPredictorAsync(Bitmap bitmap, FaceMarkPoint[] points)
            => await Task.Run(() => FaceGenderPredictor(bitmap, points));

        /// <summary>
        /// 眼睛状态检测。
        /// <para>
        /// <see cref="FaceEyeStateDetector(Bitmap, FaceMarkPoint[])"/> 的异步版本。<br />
        /// 眼睛的左右是相对图片内容而言的左右 <br />
        /// 需要模型 <see langword="eye_state.csta"/> 
        /// </para>
        /// </summary>
        /// <param name="bitmap">待识别的图像</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <returns></returns>
        public async Task<EyeStateResult> FaceEyeStateDetectorAsync(Bitmap bitmap, FaceMarkPoint[] points)
            => await Task.Run(() => FaceEyeStateDetector(bitmap, points));
    }
}

#endif