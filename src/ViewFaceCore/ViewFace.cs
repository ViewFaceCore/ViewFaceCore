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
    public sealed class ViewFace : IViewFace
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
        /// 获取或设置质量评估器的配置
        /// </summary>
        public QualityConfig QualityConfig { get; set; } = new QualityConfig();

        private readonly static object _faceDetectorLocker = new object();

        // Public Method
        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回人脸的信息。
        /// <para>
        /// 可以通过 <see cref="DetectorConfig"/> 属性对人脸检测器进行配置，以应对不同场景的图片。
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="DetectorConfig"/> 重新检测。</returns>
        public IEnumerable<FaceInfo> FaceDetector(FaceImage image)
        {
            lock (_faceDetectorLocker)
            {
                int size = 0;
                var ptr = ViewFaceNative.Detector(ref image, ref size, DetectorConfig.FaceSize, DetectorConfig.Threshold, DetectorConfig.MaxWidth, DetectorConfig.MaxHeight);
                if (ptr != IntPtr.Zero)
                {
                    for (int i = 0; i < size; i++)
                    {
                        int ofs = i * Marshal.SizeOf(typeof(FaceInfo));
                        var info = (FaceInfo)Marshal.PtrToStructure(ptr + ofs, typeof(FaceInfo));
                        yield return info;
                    }
                    ViewFaceNative.Free(ptr);
                }
            }
        }

        private readonly static object _faceMarkLocker = new object();

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
            lock (_faceMarkLocker)
            {
                long size = 0;
                var ptr = ViewFaceNative.FaceMark(ref image, info.Location, ref size, (int)MarkType);
                if (ptr != IntPtr.Zero)
                {
                    for (int i = 0; i < size; i++)
                    {
                        var ofs = i * Marshal.SizeOf(typeof(FaceMarkPoint));
                        var point = (FaceMarkPoint)Marshal.PtrToStructure(ptr + ofs, typeof(FaceMarkPoint));
                        yield return point;
                    }
                    ViewFaceNative.Free(ptr);
                }
            }
        }

        private readonly static object _extractLocker = new object();

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
        public float[] Extract(FaceImage image, IEnumerable<FaceMarkPoint> points)
        {
            lock (_extractLocker)
            {
                int size = 0;
                var ptr = ViewFaceNative.Extract(ref image, points.ToArray(), ref size, (int)FaceType);
                if (ptr != IntPtr.Zero)
                {
                    try
                    {
                        float[] result = new float[size];
                        Marshal.Copy(ptr, result, 0, size);
                        return result;
                    }
                    finally
                    {
                        ViewFaceNative.Free(ptr);
                    }
                }
                return new float[0];
            }
        }

        /// <summary>
        /// 计算特征值相似度。
        /// </summary>
        /// <param name="lfs"></param>
        /// <param name="rfs"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="ArgumentException"></exception>
        public float Compare(float[] lfs, float[] rfs)
        {
            if (lfs == null || !lfs.Any() || rfs == null || !rfs.Any())
            { throw new ArgumentNullException(nameof(lfs), "参数不能为空"); }

            if (lfs.Length != rfs.Length)
            { throw new ArgumentException("两个人脸特征值数组长度不一致，请使用同一检测模型"); }

            float sum = 0;
            for (int i = 0; i < lfs.Length; i++)
            {
                sum += lfs[i] * rfs[i];
            }
            return sum;

            //调用Native组件
            //return ViewFaceNative.Compare(_lfs, _rfs, _lfs.Length);
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
        public bool IsSelf(float[] lfs, float[] rfs) => Compare(lfs, rfs) > FaceCompareConfig.GetThreshold(FaceType);

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

        private readonly static object _disposeLocker = new object();

        public void Dispose()
        {
            lock (_disposeLocker)
            {
                ViewFaceNative.Dispose();
            }
        }
    }

}
