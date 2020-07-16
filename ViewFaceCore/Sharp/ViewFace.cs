using System;
using System.Collections.Generic;
using System.Drawing;

using ViewFaceCore.Plus;
using ViewFaceCore.Sharp.Extends;
using ViewFaceCore.Sharp.Model;

namespace ViewFaceCore.Sharp
{
    /// <summary>
    /// 人脸识别类
    /// </summary>
    public class ViewFace
    {
        bool Platform64 { get; set; } = false;
        // <para>需要模型：<see langword=""/></para>

        // ctor
        /// <summary>
        /// 使用默认的模型目录初始化人脸识别类
        /// </summary>
        public ViewFace() : this("./model/") { }
        /// <summary>
        /// 使用指定的模型目录初始化人脸识别类
        /// </summary>
        /// <param name="modelPath">模型目录</param>
        public ViewFace(string modelPath)
        {
            Platform64 = IntPtr.Size == 8;
            if (Platform64)
            { ViewFacePlus64.SetModelPath(modelPath); }
            else
            { ViewFacePlus32.SetModelPath(modelPath); }
        }
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
        public ViewFace(string modelPath, LogCallBack action) : this(modelPath)
        {
            if (Platform64)
            { ViewFacePlus64.SetLogFunction(action); }
            else
            { ViewFacePlus32.SetLogFunction(action); }
        }

        // public property
        /// <summary>
        /// 获取或设置模型路径
        /// </summary>
        public string ModelPath
        {
            get
            {
                if (Platform64)
                { return ViewFacePlus64.GetModelPath(); }
                else
                { return ViewFacePlus32.GetModelPath(); }
            }
            set
            {
                if (Platform64)
                { ViewFacePlus64.SetModelPath(value); }
                else
                { ViewFacePlus32.SetModelPath(value); }
            }
        }
        /// <summary>
        /// 获取或设置人脸类型
        /// <para>
        /// <listheader>此属性可影响到以下方法：</listheader><br />
        /// • <c><see cref="FaceDetector(Bitmap)"/></c><br />
        /// • <c><see cref="Extract(Bitmap, FaceMarkPoint[])"/></c><br />
        /// • <c><see cref="Similarity(float[], float[])"/></c><br />
        /// </para>
        /// </summary>
        public FaceType FaceType { get; set; } = FaceType.Light;
        /// <summary>
        /// 获取或设置人脸关键点类型
        /// <para>
        /// <listheader>此属性可影响到以下方法：</listheader><br />
        /// • <c><see cref="FaceMark(Bitmap, FaceInfo)"/></c><br />
        /// </para>
        /// </summary>
        public MarkType MarkType { get; set; } = MarkType.Light;
        /// <summary>
        /// 获取或设置人脸检测器设置
        /// </summary>
        public DetectorSetting DetectorSetting { get; set; } = new DetectorSetting();

        // public method
        /// <summary>
        /// 识别 <paramref name="bitmap"/> 中的人脸，并返回人脸的信息。
        /// <para>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> <see langword="||"/> <see cref="FaceType.Light"/></c> 时， 需要模型：<see langword="face_detector.csta"/><br/>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/></c> 时， 需要模型：<see langword="mask_detector.csta"/><br/>
        /// </para>
        /// </summary>
        /// <param name="bitmap">包含人脸的图片</param>
        /// <returns></returns>
        public FaceInfo[] FaceDetector(Bitmap bitmap)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            int size;
            if (Platform64)
            { size = ViewFacePlus64.DetectorSize(bgr, width, height, channels, DetectorSetting.FaceSize, DetectorSetting.Threshold, DetectorSetting.MaxWidth, DetectorSetting.MaxHeight, (int)FaceType); }
            else
            { size = ViewFacePlus32.DetectorSize(bgr, width, height, channels, DetectorSetting.FaceSize, DetectorSetting.Threshold, DetectorSetting.MaxWidth, DetectorSetting.MaxHeight, (int)FaceType); }

            if (size == -1)
            { return new FaceInfo[0]; }

            float[] _socre = new float[size];
            int[] _x = new int[size];
            int[] _y = new int[size];
            int[] _width = new int[size];
            int[] _height = new int[size];
            bool res;
            if (Platform64)
            { res = ViewFacePlus64.Detector(_socre, _x, _y, _width, _height); }
            else
            { res = ViewFacePlus32.Detector(_socre, _x, _y, _width, _height); }
            if (res)
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
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> 时， 需要模型：<see langword="face_landmarker_pts68.csta"/><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/> 时， 需要模型：<see langword="face_landmarker_mask_pts5.csta"/><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Light"/> 时， 需要模型：<see langword="face_landmarker_pts5.csta"/><br/>
        /// </para>
        /// </summary>
        /// <param name="bitmap">包含人脸的图片</param>
        /// <param name="info">指定的人脸信息</param>
        /// <returns>若失败，则返回结果 Length == 0</returns>
        public FaceMarkPoint[] FaceMark(Bitmap bitmap, FaceInfo info)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            int size;
            if (Platform64)
            { size = ViewFacePlus64.FaceMarkSize((int)MarkType); }
            else
            { size = ViewFacePlus32.FaceMarkSize((int)MarkType); }

            if (size == -1)
            { return new FaceMarkPoint[0]; }

            double[] _pointX = new double[size];
            double[] _pointY = new double[size];
            bool res;
            if (Platform64)
            { res = ViewFacePlus64.FaceMark(bgr, width, height, channels, info.Location.X, info.Location.Y, info.Location.Width, info.Location.Height, _pointX, _pointY, (int)MarkType); }
            else
            { res = ViewFacePlus32.FaceMark(bgr, width, height, channels, info.Location.X, info.Location.Y, info.Location.Width, info.Location.Height, _pointX, _pointY, (int)MarkType); }
            if (res)
            {
                List<FaceMarkPoint> points = new List<FaceMarkPoint>();
                for (int i = 0; i < size; i++)
                { points.Add(new FaceMarkPoint() { X = _pointX[i], Y = _pointY[i] }); }
                return points.ToArray();
            }
            else
            { return new FaceMarkPoint[0]; }
        }

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
        /// <returns></returns>
        public float[] Extract(Bitmap bitmap, FaceMarkPoint[] points)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            float[] features;
            if (Platform64)
            { features = new float[ViewFacePlus64.ExtractSize((int)FaceType)]; }
            else
            { features = new float[ViewFacePlus32.ExtractSize((int)FaceType)]; }


            bool res;
            if (Platform64)
            { res = ViewFacePlus64.Extract(bgr, width, height, channels, points, features, (int)FaceType); }
            else
            { res = ViewFacePlus32.Extract(bgr, width, height, channels, points, features, (int)FaceType); }
            if (res)
            { return features; }
            else
            { return new float[0]; }
        }

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


            if (Platform64)
            { return ViewFacePlus64.Similarity(leftFeatures, rightFeatures, (int)FaceType); }
            else
            { return ViewFacePlus32.Similarity(leftFeatures, rightFeatures, (int)FaceType); }
        }

        /// <summary>
        /// 判断相似度是否为同一个人。
        /// </summary>
        /// <param name="similarity">相似度</param>
        /// <returns></returns>
        public bool IsSelf(float similarity) => similarity > Face.Threshold[FaceType];

        /// <summary>
        /// 活体检测器。
        /// <para>
        /// 单帧图片，局部检测 <br />
        /// 需通过 <see cref="FaceDetector(Bitmap)"/> 获取 <paramref name="info"/> 参数<br/>
        /// 通过 <see cref="FaceMark(Bitmap, FaceInfo)"/> 获取与 <paramref name="info"/> 参数对应的 <paramref name="points"/>
        /// </para>
        /// </summary>
        /// <param name="bitmap"></param>
        /// <param name="info"></param>
        /// <param name="points"></param>
        /// <returns></returns>
        public AntiSpoofingStatus AntiSpoofing(Bitmap bitmap, FaceInfo info, FaceMarkPoint[] points) => AntiSpoofing(bitmap, info, points, false);
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
        public AntiSpoofingStatus AntiSpoofing(Bitmap bitmap, FaceInfo info, FaceMarkPoint[] points, bool global)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);


            if (Platform64)
            { return (AntiSpoofingStatus)ViewFacePlus64.AntiSpoofing(bgr, width, height, channels, info.Location.X, info.Location.Y, info.Location.Width, info.Location.Height, points, global); }
            else
            { return (AntiSpoofingStatus)ViewFacePlus32.AntiSpoofing(bgr, width, height, channels, info.Location.X, info.Location.Y, info.Location.Width, info.Location.Height, points, global); }
        }

        /// <summary>
        /// 活体检测器。
        /// <para>
        /// 视频帧图片，局部检测<br />
        /// 需通过 <see cref="FaceDetector(Bitmap)"/> 获取 <paramref name="info"/> 参数<br/>
        /// 通过 <see cref="FaceMark(Bitmap, FaceInfo)"/> 获取与 <paramref name="info"/> 参数对应的 <paramref name="points"/>
        /// </para>
        /// <para>如果返回结果为 <see cref="AntiSpoofingStatus.Detecting"/>，则说明需要继续调用此方法，传入更多的图片</para>
        /// </summary>
        /// <param name="bitmap"></param>
        /// <param name="info"></param>
        /// <param name="points"></param>
        /// <returns></returns>
        public AntiSpoofingStatus AntiSpoofingVideo(Bitmap bitmap, FaceInfo info, FaceMarkPoint[] points) => AntiSpoofingVideo(bitmap, info, points, false);
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
        public AntiSpoofingStatus AntiSpoofingVideo(Bitmap bitmap, FaceInfo info, FaceMarkPoint[] points, bool global)
        {
            byte[] bgr = bitmap.To24BGRByteArray(out int width, out int height, out int channels);

            if (Platform64)
            { return (AntiSpoofingStatus)ViewFacePlus64.AntiSpoofingVideo(bgr, width, height, channels, info.Location.X, info.Location.Y, info.Location.Width, info.Location.Height, points, global); }
            else
            { return (AntiSpoofingStatus)ViewFacePlus32.AntiSpoofingVideo(bgr, width, height, channels, info.Location.X, info.Location.Y, info.Location.Width, info.Location.Height, points, global); }
        }

        /// <summary>
        /// 释放资源
        /// </summary>
        ~ViewFace()
        {
            if (Platform64)
            { ViewFacePlus64.ViewDispose(); }
            else
            { ViewFacePlus32.ViewDispose(); }
        }
    }
}
