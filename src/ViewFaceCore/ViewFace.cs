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
    public sealed class ViewFace : IFormattable
    {
        // Constructor
        /// <summary>
        /// 使用默认的模型目录初始化人脸识别类
        /// </summary>
        public ViewFace() : this("./viewfacecore/models/") { }
        /// <summary>
        /// 使用指定的模型目录初始化人脸识别类
        /// </summary>
        /// <param name="modelPath">模型目录</param>
        public ViewFace(string modelPath) => ModelPath = modelPath;

        // Public Property
        /// <summary>
        /// 获取或设置人脸检测器配置
        /// </summary>
        public FaceDetectorConfig DetectorConfig { get; set; } = new FaceDetectorConfig();
        /// <summary>
        /// 获取或设置模型路径
        /// </summary>
        public string ModelPath { get => ViewFaceNative.GetModelPath(); set => ViewFaceNative.SetModelPath(value); }
        /// <summary>
        /// 获取或设置人脸类型。
        /// <para>
        /// <listheader>此属性可影响到以下方法：</listheader><br />
        /// • <c> <see cref="FaceDetector(FaceImage)"/> </c><br />
        /// • <c> <see cref="Extract(FaceImage, IEnumerable{FaceMarkPoint})"/> </c><br />
        /// </para>
        /// </summary>
        public FaceType FaceType { get; set; } = FaceType.Normal;
        /// <summary>
        /// 获取或设置人脸关键点类型
        /// <para>
        /// <listheader>此属性可影响到以下方法：</listheader><br />
        /// • <c><see cref="FaceMark(FaceImage, FaceInfo)"/></c><br />
        /// </para>
        /// </summary>
        public MarkType MarkType { get; set; } = MarkType.Light;
        /// <summary>
        /// 获取或设置人脸跟踪器的配置
        /// </summary>
        public FaceTrackerConfig TrackerConfig { get; set; } = new FaceTrackerConfig();
        /// <summary>
        /// 获取或设置质量评估器的配置
        /// </summary>
        public QualityConfig QualityConfig { get; set; } = new QualityConfig();

        // Public Method
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
        public IEnumerable<FaceInfo> FaceDetector(FaceImage image)
        {
            int size = 0;
            var infos = ViewFaceNative.Detector(ref image, ref size, DetectorConfig.FaceSize, DetectorConfig.Threshold, DetectorConfig.MaxWidth, DetectorConfig.MaxHeight, (int)FaceType);

            for (int i = 0; i < size; i++)
            {
                int ofs = i * Marshal.SizeOf(typeof(FaceInfo));
                var info = (FaceInfo)Marshal.PtrToStructure(infos + ofs, typeof(FaceInfo));
                yield return info;
            }
            ViewFaceNative.Free(infos);
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
        public IEnumerable<FaceMarkPoint> FaceMark(FaceImage image, FaceInfo info)
        {
            long size = 0;
            var points = ViewFaceNative.FaceMark(ref image, info.Location, ref size, (int)MarkType);
            for (int i = 0; i < size; i++)
            {
                var ofs = i * Marshal.SizeOf(typeof(FaceMarkPoint));
                var point = (FaceMarkPoint)Marshal.PtrToStructure(points + ofs, typeof(FaceMarkPoint));
                yield return point;
            }
            ViewFaceNative.Free(points);
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
        public IEnumerable<float> Extract(FaceImage image, IEnumerable<FaceMarkPoint> points)
        {
            int size = 0;
            var features = ViewFaceNative.Extract(ref image, points.ToArray(), ref size, (int)FaceType);
            for (int i = 0; i < size; i++)
            {
                var ofs = i * Marshal.SizeOf(typeof(float));
                var feature = (float)Marshal.PtrToStructure(features + ofs, typeof(float));
                yield return feature;
            }
            ViewFaceNative.Free(features);
        }

        // 不再使用 Native 的方式来计算相似度
        /// <summary>
        /// 计算特征值相似度。
        /// <para>只能计算相同 <see cref="FaceType"/> 提取出的特征值</para>
        /// <para>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_recognizer">face_recognizer.csta</a><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_recognizer_mask">face_recognizer_mask.csta</a><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Light"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_recognizer_light">face_recognizer_light.csta</a><br/>
        /// </para>
        /// </summary>
        /// <exception cref="ArgumentException"/>
        /// <exception cref="ArgumentNullException"/>
        /// <param name="lfs"></param>
        /// <param name="rfs"></param>
        /// <returns></returns>
        [Obsolete("使用 ViewFace.Compare(IEnumerable<float>, IEnumerable<float>) 代替", true)]
        public float Similarity(IEnumerable<float> lfs, IEnumerable<float> rfs) => throw new NotImplementedException();

        /// <summary>
        /// 计算特征值相似度。
        /// <para>只能计算相同 <see cref="FaceType"/> 提取出的特征值</para>
        /// </summary>
        /// <exception cref="ArgumentException"/>
        /// <exception cref="ArgumentNullException"/>
        /// <param name="lfs">特征值一</param>
        /// <param name="rfs">特征值二</param>
        /// <returns></returns>
        public float Compare(IEnumerable<float> lfs, IEnumerable<float> rfs)
        {
            if (!lfs.Any() || !rfs.Any())
            { throw new ArgumentNullException(nameof(lfs), "参数不能为空"); }

            var _lfs = lfs.ToArray();
            var _rfs = rfs.ToArray();

            if (_lfs.Length != _rfs.Length)
            { throw new ArgumentException("两个参数长度不一致"); }

            float sum = 0;
            for (int i = 0; i < _lfs.Length; i++)
            {
                sum += _lfs[i] * _rfs[i];
            }
            return sum;
        }

        /// <summary>
        /// 判断两个特征值是否为同一个人。
        /// <para>只能对比相同 <see cref="FaceType"/> 提取出的特征值</para>
        /// </summary>
        /// <exception cref="ArgumentException"/>
        /// <exception cref="ArgumentNullException"/>
        /// <param name="lfs"></param>
        /// <param name="rfs"></param>
        /// <returns></returns>
        public bool IsSelf(IEnumerable<float> lfs, IEnumerable<float> rfs) => Compare(lfs, rfs) > FaceCompareConfig.GetThreshold(FaceType);

        /// <summary>
        /// 判断相似度是否为同一个人。
        /// </summary>
        /// <param name="similarity">相似度</param>
        /// <returns></returns>
        public bool IsSelf(float similarity) => similarity > FaceCompareConfig.GetThreshold(FaceType);

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
        public AntiSpoofingStatus AntiSpoofing(FaceImage image, FaceInfo info, IEnumerable<FaceMarkPoint> points, bool global = false)
            => (AntiSpoofingStatus)ViewFaceNative.AntiSpoofing(ref image, info.Location, points.ToArray(), global);

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
        public AntiSpoofingStatus AntiSpoofingVideo(FaceImage image, FaceInfo info, IEnumerable<FaceMarkPoint> points, bool global = false)
            => (AntiSpoofingStatus)ViewFaceNative.AntiSpoofingVideo(ref image, info.Location, points.ToArray(), global);

        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回可跟踪的人脸信息。
        /// <para>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> <see langword="||"/> <see cref="FaceType.Light"/></c> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_detector">face_detector.csta</a><br/>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/></c> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.mask_detector">mask_detector.csta</a><br/>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="TrackerConfig"/> 重新检测。</returns>
        public IEnumerable<FaceTrackInfo> FaceTrack(FaceImage image)
        {
            int size = 0;
            var infos = ViewFaceNative.FaceTrack(ref image, ref size, TrackerConfig.Stable, TrackerConfig.Interval, TrackerConfig.FaceSize, TrackerConfig.Threshold, (int)FaceType);

            for (int i = 0; i < size; i++)
            {
                int ofs = i * Marshal.SizeOf(typeof(FaceTrackInfo));
                var info = (FaceTrackInfo)Marshal.PtrToStructure(infos + ofs, typeof(FaceTrackInfo));
                yield return info;
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
        public QualityResult FaceQuality(FaceImage image, FaceInfo info, IEnumerable<FaceMarkPoint> points, QualityType type)
        {
            int level = -1;
            float score = -1;

            switch (type)
            {
                case QualityType.Brightness:
                    ViewFaceNative.QualityOfBrightness(ref image, info.Location, points.ToArray(), points.Count(), ref level, ref score,
                        QualityConfig.Brightness.V0, QualityConfig.Brightness.V1, QualityConfig.Brightness.V2, QualityConfig.Brightness.V3);
                    break;
                case QualityType.Clarity:
                    ViewFaceNative.QualityOfClarity(ref image, info.Location, points.ToArray(), points.Count(), ref level, ref score, QualityConfig.Clarity.Low, QualityConfig.Clarity.High);
                    break;
                case QualityType.Integrity:
                    ViewFaceNative.QualityOfIntegrity(ref image, info.Location, points.ToArray(), points.Count(), ref level, ref score,
                        QualityConfig.Integrity.Low, QualityConfig.Integrity.High);
                    break;
                case QualityType.Pose:
                    ViewFaceNative.QualityOfPose(ref image, info.Location, points.ToArray(), points.Count(), ref level, ref score);
                    break;
                case QualityType.PoseEx:
                    ViewFaceNative.QualityOfPoseEx(ref image, info.Location, points.ToArray(), points.Count(), ref level, ref score,
                       QualityConfig.PoseEx.YawLow, QualityConfig.PoseEx.YawHigh,
                       QualityConfig.PoseEx.PitchLow, QualityConfig.PoseEx.PitchHigh,
                       QualityConfig.PoseEx.RollLow, QualityConfig.PoseEx.RollHigh);
                    break;
                case QualityType.Resolution:
                    ViewFaceNative.QualityOfResolution(ref image, info.Location, points.ToArray(), points.Count(), ref level, ref score, QualityConfig.Resolution.Low, QualityConfig.Resolution.High);
                    break;
                case QualityType.ClarityEx:
                    ViewFaceNative.QualityOfClarityEx(ref image, info.Location, points.ToArray(), points.Count(), ref level, ref score, QualityConfig.ClarityEx.BlurThresh);
                    break;
                case QualityType.Structure:
                    ViewFaceNative.QualityOfNoMask(ref image, info.Location, points.ToArray(), points.Count(), ref level, ref score);
                    break;
            }

            return new QualityResult() { Level = (QualityLevel)level, Score = score };
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
        public int FaceAgePredictor(FaceImage image, IEnumerable<FaceMarkPoint> points)
        {
            return ViewFaceNative.AgePredictor(ref image, points.ToArray(), points.Count());
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
        public Gender FaceGenderPredictor(FaceImage image, IEnumerable<FaceMarkPoint> points)
        {
            return (Gender)ViewFaceNative.GenderPredictor(ref image, points.ToArray(), points.Count());
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
        public EyeStateResult FaceEyeStateDetector(FaceImage image, IEnumerable<FaceMarkPoint> points)
        {
            int left_eye = 0, right_eye = 0;
            ViewFaceNative.EyeStateDetector(ref image, points.ToArray(), points.Count(), ref left_eye, ref right_eye);
            return new EyeStateResult((EyeState)left_eye, (EyeState)right_eye);
        }


        #region IFormattable
        /// <summary>
        /// 返回可视化字符串
        /// </summary>
        /// <returns></returns>
        public override string ToString() => ToString(null, null);
        /// <summary>
        /// 返回可视化字符串
        /// </summary>
        /// <param name="format"></param>
        /// <returns></returns>
        public string ToString(string format) => ToString(format, null);
        /// <summary>
        /// 返回可视化字符串
        /// </summary>
        /// <param name="format"></param>
        /// <param name="formatProvider"></param>
        /// <returns></returns>
        public string ToString(string format, IFormatProvider formatProvider)
        {
            string mtips = nameof(ModelPath), otips = "OperatingSystem", atips = "ProcessArchitecture";

            if ((formatProvider ?? Thread.CurrentThread.CurrentCulture) is CultureInfo cultureInfo && cultureInfo.Name.StartsWith("zh"))
            { mtips = "模型路径"; otips = "操作系统"; atips = "进程架构"; }

            return $"{{{mtips}:{ModelPath}, {otips}:{RuntimeInformation.OSDescription}, {atips}:{RuntimeInformation.ProcessArchitecture}}}";
        }
        #endregion
    }

}
