using System;
using System.Collections.Generic;
using System.Drawing;
using ViewFaceCore.Plus;
using ViewFaceCore.Sharp.Model;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace ViewFaceCore.Sharp
{
    class ImageSet
    {
        /// <summary>
        /// 获取 <see cref="Bitmap"/> 的 8bit BGR <see cref="byte"/> 数组。
        /// </summary>
        /// <param name="bitmap">待转换图像</param>
        /// <param name="width">图像宽度</param>
        /// <param name="height">图像高度</param>
        /// <param name="channels">图像扫描宽度</param>
        /// <returns></returns>
        public static byte[] Get24BGRFromBitmap(Bitmap bitmap, out int width, out int height, out int channels)
        {
            Bitmap bmp = (Bitmap)bitmap.Clone();
            if (bitmap.PixelFormat != PixelFormat.Format24bppRgb)
            {
                bmp = new Bitmap(bitmap.Width, bitmap.Height, PixelFormat.Format24bppRgb);
                using (Graphics g = Graphics.FromImage(bmp))
                {
                    g.DrawImage(bitmap, new Rectangle(0, 0, bitmap.Width, bitmap.Height));
                }
            }
            if (bmp.HorizontalResolution != 96 || bmp.VerticalResolution != 96)
            { bmp.SetResolution(96, 96); }
            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            int bytesLength = bmp.Width * bmp.Height * 3; // Get the BGR values's length.
            width = bmp.Width;
            height = bmp.Height;
            channels = 3;
            byte[] buffer = new byte[bytesLength];
            // Copy the BGR values into the array.
            Marshal.Copy(bmpData.Scan0, buffer, 0, bytesLength);
            bmp.UnlockBits(bmpData);
            bmp.Dispose();
            return buffer;
        }
    }

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
        /// 获取或设置人脸类型
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
            byte[] bgr = ImageSet.Get24BGRFromBitmap(bitmap, out int width, out int height, out int channels);
            int size;
            if (Platform64)
            { size = ViewFacePlus64.DetectorSize(bgr, width, height, channels, DetectorSetting.FaceSize, DetectorSetting.Threshold, DetectorSetting.MaxWidth, DetectorSetting.MaxHeight, (int)FaceType); }
            else
            { size = ViewFacePlus32.DetectorSize(bgr, width, height, channels, DetectorSetting.FaceSize, DetectorSetting.Threshold, DetectorSetting.MaxWidth, DetectorSetting.MaxHeight, (int)FaceType); }
            float[] _socre = new float[size];
            int[] _x = new int[size];
            int[] _y = new int[size];
            int[] _width = new int[size];
            int[] _height = new int[size];
            if (Platform64)
            { _ = ViewFacePlus64.Detector(_socre, _x, _y, _width, _height); }
            else
            { _ = ViewFacePlus32.Detector(_socre, _x, _y, _width, _height); }
            List<FaceInfo> infos = new List<FaceInfo>();
            for (int i = 0; i < size; i++)
            {
                infos.Add(new FaceInfo() { Score = _socre[i], Location = new FaceRect() { X = _x[i], Y = _y[i], Width = _width[i], Height = _height[i] } });
            }
            return infos.ToArray();
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
        /// <returns></returns>
        public FaceMarkPoint[] FaceMark(Bitmap bitmap, FaceInfo info)
        {
            byte[] bgr = ImageSet.Get24BGRFromBitmap(bitmap, out int width, out int height, out int channels);
            int size;
            if (Platform64)
            { size = ViewFacePlus64.FaceMarkSize((int)MarkType); }
            else
            { size = ViewFacePlus32.FaceMarkSize((int)MarkType); }
            double[] _pointX = new double[size];
            double[] _pointY = new double[size];
            bool val;
            if (Platform64)
            { val = ViewFacePlus64.FaceMark(bgr, width, height, channels, info.Location.X, info.Location.Y, info.Location.Width, info.Location.Height, _pointX, _pointY, (int)MarkType); }
            else
            { val = ViewFacePlus32.FaceMark(bgr, width, height, channels, info.Location.X, info.Location.Y, info.Location.Width, info.Location.Height, _pointX, _pointY, (int)MarkType); }
            if (val)
            {
                List<FaceMarkPoint> points = new List<FaceMarkPoint>();
                for (int i = 0; i < size; i++)
                { points.Add(new FaceMarkPoint() { X = _pointX[i], Y = _pointY[i] }); }
                return points.ToArray();
            }
            else
            { throw new Exception("人脸关键点获取失败"); }
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
            byte[] bgr = ImageSet.Get24BGRFromBitmap(bitmap, out int width, out int height, out int channels);
            float[] features;
            if (Platform64)
            { features = new float[ViewFacePlus64.ExtractSize((int)FaceType)]; }
            else
            { features = new float[ViewFacePlus32.ExtractSize((int)FaceType)]; }

            if (Platform64)
            { ViewFacePlus64.Extract(bgr, width, height, channels, points, features, (int)FaceType); }
            else
            { ViewFacePlus32.Extract(bgr, width, height, channels, points, features, (int)FaceType); }
            return features;
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
        /// 判断相似度是否为同一个人
        /// </summary>
        /// <param name="similarity">相似度</param>
        /// <returns></returns>
        public bool IsSelf(float similarity) => similarity > Face.Threshold[FaceType];

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
