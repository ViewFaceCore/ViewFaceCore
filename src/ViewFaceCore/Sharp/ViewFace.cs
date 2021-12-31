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
    /// <summary>
    /// 日志回调函数
    /// </summary>
    /// <param name="logText"></param>
    public delegate void LogCallBack(string logText);

    /// <summary>
    /// 人脸识别类
    /// </summary>
    public sealed class ViewFace : IDisposable
    {
        // Constructor
        /// <summary>
        /// 使用默认的模型目录初始化人脸识别类
        /// </summary>
        public ViewFace() : this("./models/") { }
        /// <summary>
        /// 使用指定的模型目录初始化人脸识别类
        /// </summary>
        /// <param name="modelPath">模型目录</param>
        public ViewFace(string modelPath) => ModelPath = modelPath;
        /// <summary>
        /// 使用指定的日志回调函数初始化人脸识别类
        /// </summary>
        /// <param name="action">日志回调函数</param>
        public ViewFace(LogCallBack action) : this("./models/", action) { }
        /// <summary>
        /// 使用指定的模型目录、日志回调函数初始化人脸识别类
        /// </summary>
        /// <param name="modelPath">模型目录</param>
        /// <param name="action">日志回调函数</param>
        public ViewFace(string modelPath, LogCallBack action) : this(modelPath) => ViewFaceBridge.SetLogFunction(action);

        /// <summary>
        /// 显示 ViewFace 当前运行的处理器架构。
        /// </summary>
        /// <returns></returns>
        public override string ToString() => $"处理器架构:{(Environment.Is64BitProcess ? "x64" : "x86")} {base.ToString()}";

        // Public Property
        /// <summary>
        /// 获取或设置人脸检测器配置
        /// </summary>
        public FaceDetectorConfig DetectorConfig { get; set; } = new FaceDetectorConfig();
        /// <summary>
        /// 获取或设置模型路径
        /// </summary>
        public string ModelPath { get => ViewFaceBridge.ModelPath; set => ViewFaceBridge.ModelPath = value; }
        /// <summary>
        /// 获取或设置人脸类型。
        /// <para>
        /// <listheader>此属性可影响到以下方法：</listheader><br />
        /// • <c><see cref="FaceDetector"/></c><br />
        /// • <c><see cref="Extract"/></c><br />
        /// • <c><see cref="Similarity"/></c><br />
        /// • <c><see cref="IsSelf" /></c><br />
        /// </para>
        /// </summary>
        public FaceType FaceType { get; set; } = FaceType.Normal;
        /// <summary>
        /// 获取或设置人脸关键点类型
        /// <para>
        /// <listheader>此属性可影响到以下方法：</listheader><br />
        /// • <c><see cref="FaceMark(Bitmap, FaceInfo)"/></c><br />
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
        /// 识别 <paramref name="bitmap"/> 中的人脸，并返回人脸的信息。
        /// <para>
        /// 可以通过 <see cref="DetectorConfig"/> 属性对人脸检测器进行配置，以应对不同场景的图片。
        /// </para>
        /// <para>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> <see langword="||"/> <see cref="FaceType.Light"/></c> 时， 需要模型： <a href="https://www.nuget.org/packages/ViewFaceCore.model.face_detector/">face_detector.csta</a><br/>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/></c> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.mask_detector">mask_detector.csta</a><br/>
        /// </para>
        /// </summary>
        /// <param name="bitmap">包含人脸的图片</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="DetectorConfig"/> 重新检测。</returns>
        public FaceInfo[] FaceDetector(Bitmap bitmap)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            FaceImage img = new FaceImage(width, height, channels);
            int size = ViewFaceBridge.DetectorSize(bgr, ref img, DetectorConfig.FaceSize, DetectorConfig.Threshold, DetectorConfig.MaxWidth, DetectorConfig.MaxHeight, (int)FaceType);

            if (size == -1)
            { return new FaceInfo[0]; }

            float[] _socre = new float[size];
            int[] _x = new int[size];
            int[] _y = new int[size];
            int[] _width = new int[size];
            int[] _height = new int[size];

            if (ViewFaceBridge.Detector(_socre, _x, _y, _width, _height))
            {
                List<FaceInfo> infos = new List<FaceInfo>();
                for (int i = 0; i < size; i++)
                {
                    infos.Add(new FaceInfo() { Score = _socre[i], Location = new FaceRect() { X = _x[i], Y = _y[i], Width = _width[i], Height = _height[i] } });
                }
                return infos.ToArray();
            }
            else { return new FaceInfo[0]; }
        }

        /// <summary>
        /// 识别 <paramref name="bitmap"/> 中指定的人脸信息 <paramref name="info"/> 的关键点坐标。
        /// <para>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_pts68">face_landmarker_pts68.csta</a><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_mask_pts5">face_landmarker_mask_pts5.csta</a><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Light"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_pts5">face_landmarker_pts5.csta</a><br/>
        /// </para>
        /// </summary>
        /// <param name="bitmap">包含人脸的图片</param>
        /// <param name="info">指定的人脸信息</param>
        /// <exception cref="MarkException"/>
        /// <returns>若失败，则返回结果 Length == 0</returns>
        public FaceMarkPoint[] FaceMark(Bitmap bitmap, FaceInfo info)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            int size = ViewFaceBridge.FaceMarkSize((int)MarkType);

            if (size == -1)
            { return new FaceMarkPoint[0]; }

            double[] _pointX = new double[size];
            double[] _pointY = new double[size];

            FaceImage img = new FaceImage(width, height, channels);
            if (ViewFaceBridge.FaceMark(bgr, ref img, info.Location, _pointX, _pointY, (int)MarkType))
            {
                List<FaceMarkPoint> points = new List<FaceMarkPoint>();
                for (int i = 0; i < size; i++)
                { points.Add(new FaceMarkPoint() { X = _pointX[i], Y = _pointY[i] }); }
                return points.ToArray();
            }
            else
            { throw new MarkException("人脸关键点获取失败"); }
        }

        /// <summary>
        /// 提取人脸特征值。
        /// <para>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_recognizer">face_recognizer.csta</a><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_recognizer_mask">face_recognizer_mask.csta</a><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Light"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_recognizer_light">face_recognizer_light.csta</a><br/>
        /// </para>
        /// </summary>
        /// <param name="bitmap"></param>
        /// <param name="points"></param>
        /// <exception cref="ExtractException"/>
        /// <returns></returns>
        public float[] Extract(Bitmap bitmap, FaceMarkPoint[] points)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            float[] features = new float[ViewFaceBridge.ExtractSize((int)FaceType)];

            FaceImage img = new FaceImage(width, height, channels);
            if (ViewFaceBridge.Extract(bgr, ref img, points, features, (int)FaceType))
            { return features; }
            else
            { throw new ExtractException("人脸特征值提取失败"); }
        }

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
        /// <param name="leftFeatures"></param>
        /// <param name="rightFeatures"></param>
        /// <returns></returns>
        public float Similarity(float[] leftFeatures, float[] rightFeatures)
        {
            if (leftFeatures.Length == 0 || rightFeatures.Length == 0)
                throw new ArgumentNullException("参数不能为空", nameof(leftFeatures));
            if (leftFeatures.Length != rightFeatures.Length)
                throw new ArgumentException("两个参数长度不一致");

            return ViewFaceBridge.Similarity(leftFeatures, rightFeatures, (int)FaceType);
        }

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
        /// <param name="bitmap">待检测的图片</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector(Bitmap)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceMark(Bitmap, FaceInfo)"/> 获取</para></param>
        /// <param name="global">是否启用全局检测能力</param>
        /// <returns>活体检测状态</returns>
        public AntiSpoofingStatus AntiSpoofing(Bitmap bitmap, FaceInfo info, FaceMarkPoint[] points, bool global = false)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);

            FaceImage img = new FaceImage(width, height, channels);
            return (AntiSpoofingStatus)ViewFaceBridge.AntiSpoofing(bgr, ref img, info.Location, points, global);
        }

        /// <summary>
        /// 活体检测器。(视频帧图片)
        /// <para>
        /// 当 <paramref name="global"/> <see langword="= false"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.fas_first">fas_first.csta</a><br/>
        /// 当 <paramref name="global"/> <see langword="= true"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.fas_second">fas_second.csta</a>
        /// </para>
        /// </summary>
        /// <param name="bitmap">待检测的图片</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector(Bitmap)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceMark(Bitmap, FaceInfo)"/> 获取</para></param>
        /// <param name="global">是否启用全局检测能力</param>
        /// <returns>如果为 <see cref="AntiSpoofingStatus.Detecting"/>，则说明需要继续调用此方法，传入更多的图片</returns>
        public AntiSpoofingStatus AntiSpoofingVideo(Bitmap bitmap, FaceInfo info, FaceMarkPoint[] points, bool global = false)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);

            FaceImage img = new FaceImage(width, height, channels);
            return (AntiSpoofingStatus)ViewFaceBridge.AntiSpoofingVideo(bgr, ref img, info.Location, points, global);
        }

        /// <summary>
        /// 识别 <paramref name="bitmap"/> 中的人脸，并返回可跟踪的人脸信息。
        /// <para>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> <see langword="||"/> <see cref="FaceType.Light"/></c> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_detector">face_detector.csta</a><br/>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/></c> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.mask_detector">mask_detector.csta</a><br/>
        /// </para>
        /// </summary>
        /// <param name="bitmap">待检测的图片</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="TrackerConfig"/> 重新检测。</returns>
        public FaceTrackInfo[] FaceTrack(Bitmap bitmap)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            FaceImage img = new FaceImage(width, height, channels);
            int size = ViewFaceBridge.FaceTrackSize(bgr, ref img, TrackerConfig.Stable, TrackerConfig.Interval, TrackerConfig.FaceSize, TrackerConfig.Threshold, (int)FaceType);

            if (size == -1)
            { return new FaceTrackInfo[0]; }

            float[] _socre = new float[size];
            int[] _pid = new int[size];
            int[] _x = new int[size];
            int[] _y = new int[size];
            int[] _width = new int[size];
            int[] _height = new int[size];

            if (ViewFaceBridge.FaceTrack(_socre, _pid, _x, _y, _width, _height))
            {
                List<FaceTrackInfo> infos = new List<FaceTrackInfo>();
                for (int i = 0; i < size; i++)
                {
                    infos.Add(new FaceTrackInfo() { Score = _socre[i], Pid = _pid[i], Location = new FaceRect() { X = _x[i], Y = _y[i], Width = _width[i], Height = _height[i] } });
                }
                return infos.ToArray();
            }
            else { return new FaceTrackInfo[0]; }
        }

        /// <summary>
        /// 人脸质量评估
        /// </summary>
        /// <param name="bitmap">待检测的图片</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector(Bitmap)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceMark(Bitmap, FaceInfo)"/> 获取</para></param>
        /// <param name="type">质量评估类型</param>
        /// <returns></returns>
        public QualityResult FaceQuality(Bitmap bitmap, FaceInfo info, FaceMarkPoint[] points, QualityType type)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            FaceImage img = new FaceImage(width, height, channels);
            int level = 0; float score = 0; bool res = false;

            switch (type)
            {
                case QualityType.Brightness:
                    res = ViewFaceBridge.QualityOfBrightness(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score,
                        QualityConfig.Brightness.V0, QualityConfig.Brightness.V1, QualityConfig.Brightness.V2, QualityConfig.Brightness.V3);
                    break;
                case QualityType.Clarity:
                    res = ViewFaceBridge.QualityOfClarity(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score,
                        QualityConfig.Clarity.Low, QualityConfig.Clarity.High);
                    break;
                case QualityType.Integrity:
                    res = ViewFaceBridge.QualityOfIntegrity(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score,
                        QualityConfig.Integrity.Low, QualityConfig.Integrity.High);
                    break;
                case QualityType.Pose:
                    res = ViewFaceBridge.QualityOfPose(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score);
                    break;
                case QualityType.PoseEx:
                    res = ViewFaceBridge.QualityOfPoseEx(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score,
                        QualityConfig.PoseEx.YawLow, QualityConfig.PoseEx.YawHigh,
                        QualityConfig.PoseEx.PitchLow, QualityConfig.PoseEx.PitchHigh,
                        QualityConfig.PoseEx.RollLow, QualityConfig.PoseEx.RollHigh);
                    break;
                case QualityType.Resolution:
                    res = ViewFaceBridge.QualityOfResolution(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score,
                        QualityConfig.Resolution.Low, QualityConfig.Resolution.High);
                    break;
                case QualityType.ClarityEx:
                    res = ViewFaceBridge.QualityOfClarityEx(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score,
                        QualityConfig.ClarityEx.BlurThresh);
                    break;
                case QualityType.Structure:
                    res = ViewFaceBridge.QualityOfNoMask(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score);
                    break;
            }

            if (res)
            { return new QualityResult() { Level = (QualityLevel)level, Score = score }; }
            else
            { return new QualityResult() { Level = QualityLevel.Error, Score = -1 }; }
        }


        /// <summary>
        /// 年龄预测。
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.age_predictor">age_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="bitmap">待检测的图片</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="FaceMark(Bitmap, FaceInfo)"/> 获取</para></param>
        /// <returns>-1: 预测失败失败，其它: 预测的年龄。</returns>
        public int FaceAgePredictor(Bitmap bitmap, FaceMarkPoint[] points)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            FaceImage img = new FaceImage(width, height, channels);
            return ViewFaceBridge.AgePredictor(bgr, ref img, points, points.Length);
        }

        /// <summary>
        /// 性别预测。
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.gender_predictor">gender_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="bitmap">待检测的图片</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="FaceMark(Bitmap, FaceInfo)"/> 获取</para></param>
        /// <returns>性别。<see cref="Gender.Unknown"/> 代表识别失败</returns>
        public Gender FaceGenderPredictor(Bitmap bitmap, FaceMarkPoint[] points)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            FaceImage img = new FaceImage(width, height, channels);
            return (Gender)ViewFaceBridge.GenderPredictor(bgr, ref img, points, points.Length);
        }

        /// <summary>
        /// 眼睛状态检测。
        /// <para>
        /// 眼睛的左右是相对图片内容而言的左右。<br />
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.eye_state">eye_state.csta</a>
        /// </para>
        /// </summary>
        /// <param name="bitmap">待检测的图片</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="FaceMark(Bitmap, FaceInfo)"/> 获取</para></param>
        /// <returns></returns>
        public EyeStateResult FaceEyeStateDetector(Bitmap bitmap, FaceMarkPoint[] points)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            FaceImage img = new FaceImage(width, height, channels);
            int left_eye = 0, right_eye = 0;
            ViewFaceBridge.EyeStateDetector(bgr, ref img, points, points.Length, ref left_eye, ref right_eye);
            return new EyeStateResult((EyeState)left_eye, (EyeState)right_eye);
        }


        private bool disposed = false;

        /// <summary>
        /// 
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (!this.disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources.
                }

                ViewFaceBridge.ViewDispose();

                disposed = true;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        ~ViewFace()
        {
            Dispose(false);
        }
    }
}
