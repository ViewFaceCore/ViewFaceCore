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
    /// 人脸识别类
    /// </summary>
    public class ViewFace
    {
        // Constructor
        /// <summary>
        /// 使用默认的模型目录初始化人脸识别类
        /// </summary>
        public ViewFace() : this("./model/") { }
        /// <summary>
        /// 使用指定的模型目录初始化人脸识别类
        /// </summary>
        /// <param name="modelPath">模型目录</param>
        public ViewFace(string modelPath) => ModelPath = modelPath;
        /// <summary>
        /// 使用指定的日志回调函数初始化人脸识别类
        /// </summary>
        /// <param name="action">日志回调函数</param>
        public ViewFace(LogCallBack action) : this("./model/", action) { }
        /// <summary>
        /// 使用指定的模型目录、日志回调函数初始化人脸识别类
        /// </summary>
        /// <param name="modelPath">模型目录</param>
        /// <param name="action">日志回调函数</param>
        public ViewFace(string modelPath, LogCallBack action) : this(modelPath) => ViewFacePlus.SetLogFunction(action);

        /// <summary>
        /// 显示 ViewFace 当前运行的 CPU 类型。
        /// </summary>
        /// <returns></returns>
        public override string ToString() => $"处理器:{(ViewFacePlus.Is64BitProcess ? "x64" : "x86")} {base.ToString()}";

        // Public Property
        /// <summary>
        /// 获取或设置人脸检测器配置
        /// </summary>
        public FaceDetectorConfig DetectorConfig { get; set; } = new FaceDetectorConfig();
        /// <summary>
        /// 获取或设置模型路径
        /// </summary>
        public string ModelPath { get => ViewFacePlus.ModelPath; set => ViewFacePlus.ModelPath = value; }
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
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> <see langword="||"/> <see cref="FaceType.Light"/></c> 时， 需要模型：<see langword="face_detector.csta"/><br/>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/></c> 时， 需要模型：<see langword="mask_detector.csta"/><br/>
        /// </para>
        /// </summary>
        /// <param name="bitmap">包含人脸的图片</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="DetectorConfig"/> 重新检测。</returns>
        public FaceInfo[] FaceDetector(Bitmap bitmap)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            FaceImage img = new FaceImage(width, height, channels);
            int size = ViewFacePlus.DetectorSize(bgr, ref img, DetectorConfig.FaceSize, DetectorConfig.Threshold, DetectorConfig.MaxWidth, DetectorConfig.MaxHeight, (int)FaceType);

            if (size == -1)
            { return new FaceInfo[0]; }

            float[] _socre = new float[size];
            int[] _x = new int[size];
            int[] _y = new int[size];
            int[] _width = new int[size];
            int[] _height = new int[size];

            if (ViewFacePlus.Detector(_socre, _x, _y, _width, _height))
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
        public FaceMarkPoint[] FaceMark(Bitmap bitmap, FaceInfo info)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            int size = ViewFacePlus.FaceMarkSize((int)MarkType);

            if (size == -1)
            { return new FaceMarkPoint[0]; }

            double[] _pointX = new double[size];
            double[] _pointY = new double[size];

            FaceImage img = new FaceImage(width, height, channels);
            if (ViewFacePlus.FaceMark(bgr, ref img, info.Location, _pointX, _pointY, (int)MarkType))
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
        public float[] Extract(Bitmap bitmap, FaceMarkPoint[] points)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            float[] features = new float[ViewFacePlus.ExtractSize((int)FaceType)];

            FaceImage img = new FaceImage(width, height, channels);
            if (ViewFacePlus.Extract(bgr, ref img, points, features, (int)FaceType))
            { return features; }
            else
            { throw new ExtractException("人脸特征值提取失败"); }
        }
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
        public float Similarity(float[] leftFeatures, float[] rightFeatures)
        {
            if (leftFeatures.Length == 0 || rightFeatures.Length == 0)
                throw new ArgumentNullException("参数不能为空", nameof(leftFeatures));
            if (leftFeatures.Length != rightFeatures.Length)
                throw new ArgumentException("两个参数长度不一致");

            return ViewFacePlus.Similarity(leftFeatures, rightFeatures, (int)FaceType);
        }
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
        /// 判断相似度是否为同一个人。
        /// </summary>
        /// <param name="similarity">相似度</param>
        /// <returns></returns>
        public bool IsSelf(float similarity) => similarity > FaceCompareConfig.GetThreshold(FaceType);

        /// <summary>
        /// 活体检测器。
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
        public AntiSpoofingStatus AntiSpoofing(Bitmap bitmap, FaceInfo info, FaceMarkPoint[] points, bool global = false)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);

            FaceImage img = new FaceImage(width, height, channels);
            return (AntiSpoofingStatus)ViewFacePlus.AntiSpoofing(bgr, ref img, info.Location, points, global);
        }
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
        public AntiSpoofingStatus AntiSpoofingVideo(Bitmap bitmap, FaceInfo info, FaceMarkPoint[] points, bool global = false)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);

            FaceImage img = new FaceImage(width, height, channels);
            return (AntiSpoofingStatus)ViewFacePlus.AntiSpoofingVideo(bgr, ref img, info.Location, points, global);
        }
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
        public FaceTrackInfo[] FaceTrack(Bitmap bitmap)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            FaceImage img = new FaceImage(width, height, channels);
            int size = ViewFacePlus.FaceTrackSize(bgr, ref img, TrackerConfig.Stable, TrackerConfig.Interval, TrackerConfig.FaceSize, TrackerConfig.Threshold, (int)FaceType);

            if (size == -1)
            { return new FaceTrackInfo[0]; }

            float[] _socre = new float[size];
            int[] _pid = new int[size];
            int[] _x = new int[size];
            int[] _y = new int[size];
            int[] _width = new int[size];
            int[] _height = new int[size];

            if (ViewFacePlus.FaceTrack(_socre, _pid, _x, _y, _width, _height))
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
        /// </summary>
        /// <param name="bitmap"></param>
        /// <param name="info"></param>
        /// <param name="points"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        public QualityResult FaceQuality(Bitmap bitmap, FaceInfo info, FaceMarkPoint[] points, QualityType type)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            FaceImage img = new FaceImage(width, height, channels);
            int level = 0; float score = 0; bool res = false;

            switch (type)
            {
                case QualityType.Brightness:
                    res = ViewFacePlus.QualityOfBrightness(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score,
                        QualityConfig.Brightness.V0, QualityConfig.Brightness.V1, QualityConfig.Brightness.V2, QualityConfig.Brightness.V3);
                    break;
                case QualityType.Clarity:
                    res = ViewFacePlus.QualityOfClarity(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score,
                        QualityConfig.Clarity.Low, QualityConfig.Clarity.High);
                    break;
                case QualityType.Integrity:
                    res = ViewFacePlus.QualityOfIntegrity(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score,
                        QualityConfig.Integrity.Low, QualityConfig.Integrity.High);
                    break;
                case QualityType.Pose:
                    res = ViewFacePlus.QualityOfPose(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score);
                    break;
                case QualityType.PoseEx:
                    res = ViewFacePlus.QualityOfPoseEx(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score,
                        QualityConfig.PoseEx.YawLow, QualityConfig.PoseEx.YawHigh,
                        QualityConfig.PoseEx.PitchLow, QualityConfig.PoseEx.PitchHigh,
                        QualityConfig.PoseEx.RollLow, QualityConfig.PoseEx.RollHigh);
                    break;
                case QualityType.Resolution:
                    res = ViewFacePlus.QualityOfResolution(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score,
                        QualityConfig.Resolution.Low, QualityConfig.Resolution.High);
                    break;
                case QualityType.ClarityEx:
                    res = ViewFacePlus.QualityOfClarityEx(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score,
                        QualityConfig.ClarityEx.BlurThresh);
                    break;
                case QualityType.Structure:
                    res = ViewFacePlus.QualityOfNoMask(bgr, ref img, info.Location, points, points.Length,
                        ref level, ref score);
                    break;
            }

            if (res)
            { return new QualityResult() { Level = (QualityLevel)level, Score = score }; }
            else
            { return new QualityResult() { Level = QualityLevel.Error, Score = -1 }; }
        }
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
        /// 需要模型 <see langword="age_predictor.csta"/> 
        /// </para>
        /// </summary>
        /// <param name="bitmap">待识别的图像</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <returns>-1 则为失败，否则为预测年龄</returns>
        public int FaceAgePredictor(Bitmap bitmap, FaceMarkPoint[] points)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            FaceImage img = new FaceImage(width, height, channels);
            return ViewFacePlus.AgePredictor(bgr, ref img, points, points.Length);
        }
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
        /// 需要模型 <see langword="gender_predictor.csta"/> 
        /// </para>
        /// </summary>
        /// <param name="bitmap">待识别的图像</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <returns>性别枚举，<see cref="Gender.Unknown"/> 代表识别失败</returns>
        public Gender FaceGenderPredictor(Bitmap bitmap, FaceMarkPoint[] points)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            FaceImage img = new FaceImage(width, height, channels);
            return (Gender)ViewFacePlus.GenderPredictor(bgr, ref img, points, points.Length);
        }
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
        /// 眼睛的左右是相对图片内容而言的左右 <br />
        /// 需要模型 <see langword="eye_state.csta"/> 
        /// </para>
        /// </summary>
        /// <param name="bitmap">待识别的图像</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <returns></returns>
        public EyeStateResult FaceEyeStateDetector(Bitmap bitmap, FaceMarkPoint[] points)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            FaceImage img = new FaceImage(width, height, channels);
            int left_eye = 0, right_eye = 0;
            ViewFacePlus.EyeStateDetector(bgr, ref img, points, points.Length, ref left_eye, ref right_eye);
            return new EyeStateResult((EyeState)left_eye, (EyeState)right_eye);
        }
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

        /// <summary>
        /// 释放资源
        /// </summary>
        ~ViewFace() => ViewFacePlus.ViewDispose();
    }
}
