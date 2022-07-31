using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using ViewFaceCore.Configs;
using ViewFaceCore.Exceptions;
using ViewFaceCore.Model;
using ViewFaceCore.Native;

namespace ViewFaceCore
{
    /// <summary>
    /// 人脸识别类
    /// </summary>
    public interface IViewFace : IFormattable, IDisposable
    {
        // Public Property
        /// <summary>
        /// 获取或设置人脸检测器配置
        /// </summary>
        public FaceDetectorConfig DetectorConfig { get; set; }

        /// <summary>
        /// 获取或设置模型路径
        /// </summary>
        public string ModelPath { get; set; }

        /// <summary>
        /// 获取或设置人脸类型。
        /// <para>
        /// <listheader>此属性可影响到以下方法：</listheader><br />
        /// • <c> <see cref="FaceDetector(FaceImage)"/> </c><br />
        /// • <c> <see cref="Extract(FaceImage, IEnumerable{FaceMarkPoint})"/> </c><br />
        /// </para>
        /// </summary>
        public FaceType FaceType { get; set; }

        /// <summary>
        /// 获取或设置人脸关键点类型
        /// <para>
        /// <listheader>此属性可影响到以下方法：</listheader><br />
        /// • <c><see cref="FaceMark(FaceImage, FaceInfo)"/></c><br />
        /// </para>
        /// </summary>
        public MarkType MarkType { get; set; }

        /// <summary>
        /// 获取或设置质量评估器的配置
        /// </summary>
        public QualityConfig QualityConfig { get; set; }

        // Public Method
        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回人脸的信息。
        /// <para>
        /// 可以通过 <see cref="DetectorConfig"/> 属性对人脸检测器进行配置，以应对不同场景的图片。
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="DetectorConfig"/> 重新检测。</returns>
        public FaceInfo[] FaceDetector(FaceImage image);

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
        public FaceMarkPoint[] FaceMark(FaceImage image, FaceInfo info);

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
        public float[] Extract(FaceImage image, FaceMarkPoint[] points);

        /// <summary>
        /// 计算特征值相似度。
        /// </summary>
        /// <param name="lfs"></param>
        /// <param name="rfs"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="ArgumentException"></exception>
        public float Compare(float[] lfs, float[] rfs);

        /// <summary>
        /// 判断两个特征值是否为同一个人。
        /// <para>只能对比相同 <see cref="FaceType"/> 提取出的特征值</para>
        /// </summary>
        /// <exception cref="ArgumentException"/>
        /// <exception cref="ArgumentNullException"/>
        /// <param name="lfs"></param>
        /// <param name="rfs"></param>
        /// <returns></returns>
        public bool IsSelf(float[] lfs, float[] rfs);

        /// <summary>
        /// 判断相似度是否为同一个人。
        /// </summary>
        /// <param name="similarity">相似度</param>
        /// <returns></returns>
        public bool IsSelf(float similarity);

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
        public AntiSpoofingStatus AntiSpoofing(FaceImage image, FaceInfo info, FaceMarkPoint[] points, bool global = false);

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
        public AntiSpoofingStatus AntiSpoofingVideo(FaceImage image, FaceInfo info, FaceMarkPoint[] points, bool global = false);

        /// <summary>
        /// 人脸质量评估
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector(FaceImage)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <param name="type">质量评估类型</param>
        /// <returns></returns>
        public QualityResult FaceQuality(FaceImage image, FaceInfo info, FaceMarkPoint[] points, QualityType type);

        /// <summary>
        /// 年龄预测。
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.age_predictor">age_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>-1: 预测失败失败，其它: 预测的年龄。</returns>
        public int FaceAgePredictor(FaceImage image, FaceMarkPoint[] points);

        /// <summary>
        /// 性别预测。
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.gender_predictor">gender_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>性别。<see cref="Gender.Unknown"/> 代表识别失败</returns>
        public Gender FaceGenderPredictor(FaceImage image, FaceMarkPoint[] points);

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
        public EyeStateResult FaceEyeStateDetector(FaceImage image, FaceMarkPoint[] points);
    }

}
