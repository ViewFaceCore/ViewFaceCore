using System;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Text;
using ViewFaceCore.Configs.Enums;
using ViewFaceCore.Exceptions;
using ViewFaceCore.Models;

namespace ViewFaceCore.Native
{
    /// <summary>
    /// 导入方法
    /// </summary>
    internal static partial class ViewFaceNative
    {
        /// <summary>
        /// 依赖库默认名称
        /// </summary>
        const string BRIDGE_LIBRARY_NAME = "ViewFaceBridge";

        /// <summary>
        /// 模型路径支持最大长度
        /// </summary>
        const int MAX_PATH_LENGTH = 1024;

        #region Common

        /// <summary>
        /// 设置人脸模型的目录（Windows）
        /// </summary>
        /// <param name="path"></param>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "SetModelPath", CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        private extern static void SetModelPathWindows(string path);

        /// <summary>
        /// 设置人脸模型的目录（Linux）
        /// </summary>
        /// <param name="path"></param>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "SetModelPath", CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        private extern static void SetModelPathLinux(byte[] path);

        public static void SetModelPath(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentNullException(nameof(path), "Model path can not null.");
            //to utf-8
            byte[] pathUtf8Bytes = Encoding.Convert(Encoding.Default, Encoding.UTF8, Encoding.Default.GetBytes(path));
            if (pathUtf8Bytes.Length > MAX_PATH_LENGTH)
                throw new NotSupportedException($"The path is too long, not support path more than {MAX_PATH_LENGTH} byte.");
            path = Encoding.UTF8.GetString(pathUtf8Bytes);

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                SetModelPathWindows(path);
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                SetModelPathLinux(Encoding.UTF8.GetBytes(path));
            else
                throw new PlatformNotSupportedException($"Unsupported system type: {RuntimeInformation.OSDescription}");

            if (!path.Equals(GetModelPath()))
                throw new SeetaFaceModelException($"Set model path to '{path}' failed, failed to verify this path.");
        }

        private static string _modelPath = null;

        /// <summary>
        /// 获取人脸模型的目录
        /// </summary>
        /// <param name="outPath">获取到的路径</param>
        /// <param name="size">字符串长度</param>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "GetModelPath", CharSet = CharSet.Auto, CallingConvention = CallingConvention.Cdecl)]
        private extern static void GetModelPath(StringBuilder outPath, ref int size);
        public static string GetModelPath()
        {
            if (!string.IsNullOrWhiteSpace(_modelPath))
            {
                return _modelPath;
            }
            StringBuilder result = new StringBuilder(MAX_PATH_LENGTH);
            int size = 0;
            GetModelPath(result, ref size);
            if (size > MAX_PATH_LENGTH)
            {
                throw new NotSupportedException($"The path is too long, not support path more than {MAX_PATH_LENGTH} byte.");
            }
            _modelPath = result?.ToString();
            return _modelPath;
        }

        /// <summary>
        /// 释放本机代码中由 malloc 分配的内存。
        /// </summary>
        /// <param name="address"></param>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "Free", CallingConvention = CallingConvention.Cdecl)]
        public static extern void Free(IntPtr address);

        #endregion

        #region FaceDetector(人脸检测)

        /// <summary>
        /// 获取人脸检测句柄
        /// </summary>
        /// <param name="faceSize">最小人脸是人脸检测器常用的一个概念，默认值为20，单位像素。
        /// <para>最小人脸和检测器性能息息相关。主要方面是速度，使用建议上，我们建议在应用范围内，这个值设定的越大越好。SeetaFace采用的是BindingBox Regresion的方式训练的检测器。如果最小人脸参数设置为80的话，从检测能力上，可以将原图缩小的原来的1/4，这样从计算复杂度上，能够比最小人脸设置为20时，提速到16倍。</para>
        /// </param>
        /// <param name="threshold">检测器阈值默认值是0.9，合理范围为[0, 1]。这个值一般不进行调整，除了用来处理一些极端情况。这个值设置的越小，漏检的概率越小，同时误检的概率会提高</param>
        /// <param name="maxWidth">可检测的图像最大宽度。默认值2000。</param>
        /// <param name="maxHeight">可检测的图像最大高度。默认值2000。</param>
        /// <param name="deviceType">设备类型</param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "GetFaceDetectorHandler", CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr GetFaceDetectorHandler(double faceSize = 20, double threshold = 0.9, double maxWidth = 2000, double maxHeight = 2000, int deviceType = 0);

        /// <summary>
        /// 人脸检测器
        /// </summary>
        /// <param name="handler"></param>
        /// <param name="img">图像信息</param>
        /// <param name="size">检测到的人脸数量</param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "FaceDetectV2", CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr FaceDetectV2(IntPtr handler, ref FaceImage img, ref int size);

        /// <summary>
        /// 释放人脸检测句柄
        /// </summary>
        /// <param name="handler"></param>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "DisposeFaceDetector", CallingConvention = CallingConvention.Cdecl)]
        public extern static void DisposeFaceDetector(IntPtr handler);

        #endregion

        #region MaskDetector（口罩识别）

        /// <summary>
        /// 获取口罩识别句柄
        /// </summary>
        /// <param name="faceSize"></param>
        /// <param name="threshold"></param>
        /// <param name="maxWidth"></param>
        /// <param name="maxHeight"></param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "GetMaskDetectorHandler", CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr GetMaskDetectorHandler(int deviceType = 0);

        /// <summary>
        /// 口罩检测
        /// </summary>
        /// <param name="handler"></param>
        /// <param name="img"></param>
        /// <param name="faceRect"></param>
        /// <param name="score">一般性的，score超过0.5，则认为是检测带上了口罩。</param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "MaskDetect", CallingConvention = CallingConvention.Cdecl)]
        public extern static bool MaskDetect(IntPtr handler, ref FaceImage img, FaceRect faceRect, ref float score);

        /// <summary>
        /// 释放口罩识别句柄
        /// </summary>
        /// <param name="handler"></param>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "DisposeMaskDetector", CallingConvention = CallingConvention.Cdecl)]
        public extern static void DisposeMaskDetector(IntPtr handler);

        #endregion

        #region FaceMark

        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "GetFaceLandmarkerHandler", CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr GetFaceLandmarkerHandler(int type = 0, int deviceType = 0);

        /// <summary>
        /// 获取人脸关键点
        /// <para>需要 <see cref="Free(IntPtr)"/></para>
        /// </summary>
        /// <param name="img">图像信息</param>
        /// <param name="faceRect">人脸位置信息</param>
        /// <param name="size">关键点数量</param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "FaceMark", CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr FaceMark(IntPtr handler, ref FaceImage img, FaceRect faceRect, ref long size);

        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "DisposeFaceLandmarker", CallingConvention = CallingConvention.Cdecl)]
        public extern static void DisposeFaceLandmarker(IntPtr handler);

        #endregion

        #region FaceRecognizer

        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "GetFaceRecognizerHandler", CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr GetFaceRecognizerHandler(int type = 0, int deviceType = 0);

        /// <summary>
        /// 提取人脸特征值
        /// <para>需要 <see cref="Free(IntPtr)"/></para>
        /// </summary>
        /// <param name="img">图像信息</param>
        /// <param name="size">检测到的人脸数量</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <param name="type">模型类型。0：face_recognizer；1：face_recognizer_mask；2：face_recognizer_light。</param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "FaceRecognizerExtract", CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr FaceRecognizerExtract(IntPtr handler, ref FaceImage img, FaceMarkPoint[] points, ref int size);

        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "DisposeFaceRecognizer", CallingConvention = CallingConvention.Cdecl)]
        public extern static void DisposeFaceRecognizer(IntPtr handler);

        /// <summary>
        /// 计算相似度
        /// </summary>
        /// <param name="leftFeatures"></param>
        /// <param name="rightFeatures"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "Compare", CallingConvention = CallingConvention.Cdecl)]
        public extern static float Compare(float[] lhs, float[] rhs, int size);

        #endregion

        #region FaceAntiSpoofing（活体检测）

        /// <summary>
        /// 获取活体检测器句柄
        /// </summary>
        /// <param name="global">是否启用全局检测</param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "GetFaceAntiSpoofingHandler", CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr GetFaceAntiSpoofingHandler(int videoFrameCount = 10
            , float boxThresh = 0.8f
            , float clarity = 0.3f
            , float reality = 0.8f
            , bool global = false
            , int deviceType = 0);

        /// <summary>
        /// 活体检测器
        /// <para>单帧检测</para>
        /// </summary>
        /// <param name="img">图像宽高通道信息</param>
        /// <param name="faceRect">人脸位置信息</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <param name="global">是否启用全局检测</param>
        /// <returns>单帧识别返回值会是 <see cref="AntiSpoofingStatus.Real"/>、<see cref="AntiSpoofingStatus.Spoof"/> 或 <see cref="AntiSpoofingStatus.Fuzzy"/></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "FaceAntiSpoofingPredict", CallingConvention = CallingConvention.Cdecl)]
        public extern static int FaceAntiSpoofingPredict(IntPtr handler, ref FaceImage img, FaceRect faceRect, FaceMarkPoint[] points, ref float clarity, ref float reality);

        /// <summary>
        /// 活体检测器
        /// <para>视频帧</para>
        /// </summary>
        /// <param name="img">图像宽高通道信息</param>
        /// <param name="faceRect">人脸位置信息</param>
        /// <param name="points"></param>
        /// <param name="global">是否启用全局检测</param>
        /// <returns>
        /// <para>
        /// 单帧识别返回值会是 <see cref="AntiSpoofingStatus.Real"/>、<see cref="AntiSpoofingStatus.Spoof"/>、<see cref="AntiSpoofingStatus.Fuzzy"/> 或 <see cref="AntiSpoofingStatus.Detecting"/><br />
        /// 在视频识别输入帧数不满足需求的时候，返回状态就是 <see cref="AntiSpoofingStatus.Detecting"/>
        /// </para>
        /// </returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "FaceAntiSpoofingPredictVideo", CallingConvention = CallingConvention.Cdecl)]
        public extern static int FaceAntiSpoofingPredictVideo(IntPtr handler, ref FaceImage img, FaceRect faceRect, FaceMarkPoint[] pointsref, ref float clarity, ref float reality);

        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "DisposeFaceAntiSpoofing", CallingConvention = CallingConvention.Cdecl)]
        public extern static void DisposeFaceAntiSpoofing(IntPtr handler);

        #endregion

        #region FaceTrack

        /// <summary>
        /// 获取人脸跟踪句柄
        /// </summary>
        /// <param name="width">图像宽度</param>
        /// <param name="height">图像高度</param>
        /// <param name="type">模型类型。0：face_detector；1：mask_detector；</param>
        /// <param name="stable"></param>
        /// <param name="interval"></param>
        /// <param name="faceSize"></param>
        /// <param name="threshold"></param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "GetFaceTrackerHandler", CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr GetFaceTrackerHandler(int width, int height, bool stable = false, int interval = 10, int faceSize = 20, float threshold = 0.9f, int deviceType = 0);

        /// <summary>
        /// 人脸跟踪信息
        /// </summary>
        /// <param name="faceTracker">人脸跟踪句柄</param>
        /// <param name="img">追踪图像</param>
        /// <param name="size"></param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "FaceTrack", CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr FaceTrack(IntPtr faceTracker, ref FaceImage img, ref int size);

        /// <summary>
        /// 重置追踪视频
        /// </summary>
        /// <param name="faceTracker"></param>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "FaceTrackReset", CallingConvention = CallingConvention.Cdecl)]
        public extern static void FaceTrackReset(IntPtr faceTracker);

        /// <summary>
        /// 释放人脸追踪句柄
        /// </summary>
        /// <param name="faceTracker"></param>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "DisposeFaceTracker", CallingConvention = CallingConvention.Cdecl)]
        public extern static void DisposeFaceTracker(IntPtr faceTracker);

        #endregion

        #region 质量评估

        /// <summary>
        /// 亮度评估。
        /// <para>亮度评估就是评估人脸区域内的亮度值是否均匀正常，存在部分或全部的过亮和过暗都会是评价为LOW。</para>
        /// <para>
        /// 评估器会将综合的亮度从灰度值映射到level，其映射关系为： <br />
        /// • [0, v0), [v3, ~) => <see cref="QualityLevel.Low"/> <br />
        /// • [v0, v1), [v2, v3) => <see cref="QualityLevel.Medium"/> <br />
        /// • [v1, v2) => <see cref="QualityLevel.High"/> <br />
        /// </para> <br />
        /// <para><see langword="{v0, v1, v2, v3}"/> 的默认值为 <see langword="{70, 100, 210, 230}"/></para>
        /// </summary>
        /// <param name="img">图像宽高通道信息</param>
        /// <param name="faceRect">人脸位置信息</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <param name="pointsLength">人脸关键点 数组长度</param>
        /// <param name="level">存储 等级</param>
        /// <param name="score">存储 分数</param>
        /// <param name="v0"></param>
        /// <param name="v1"></param>
        /// <param name="v2"></param>
        /// <param name="v3"></param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "Quality_Brightness", CallingConvention = CallingConvention.Cdecl)]
        public extern static void QualityOfBrightness(ref FaceImage img, FaceRect faceRect, FaceMarkPoint[] points, int pointsLength, ref int level, ref float score,
            float v0 = 70, float v1 = 100, float v2 = 210, float v3 = 230);

        /// <summary>
        /// 清晰度评估。
        /// <para>清晰度这里是传统方式通过二次模糊后图像信息损失程度统计的清晰度。</para>
        /// <para>
        /// 映射关系为： <br />
        /// • [0, low) => <see cref="QualityLevel.Low"/> <br />
        /// • [low, high) => <see cref="QualityLevel.Medium"/> <br />
        /// • [high, ~) => <see cref="QualityLevel.High"/> <br />
        /// </para> <br />
        /// <para><see langword="{low, high}"/> 的默认值为 <see langword="{0.1, 0.2}"/></para>
        /// </summary>
        /// <param name="img">图像宽高通道信息</param>
        /// <param name="faceRect">人脸位置信息</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <param name="pointsLength">人脸关键点 数组长度</param>
        /// <param name="level">存储 等级</param>
        /// <param name="score">存储 分数</param>
        /// <param name="low"></param>
        /// <param name="high"></param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "Quality_Clarity", CallingConvention = CallingConvention.Cdecl)]
        public extern static void QualityOfClarity(ref FaceImage img, FaceRect faceRect, FaceMarkPoint[] points, int pointsLength, ref int level, ref float score,
            float low = 0.1f, float high = 0.2f);

        /// <summary>
        /// 完整度评估。
        /// <para>完整度评估是朴素的判断人来是否因为未完全进入摄像头而造成的不完整的情况。该方法不适用于判断遮挡造成的不完整。</para>
        /// <para>
        /// 映射关系为： <br />
        /// • 人脸外扩 high 倍数没有超出图像 => <see cref="QualityLevel.High"/> <br />
        /// • 人脸外扩 low 倍数没有超出图像 => <see cref="QualityLevel.Medium"/> <br />
        /// • 其他 => <see cref="QualityLevel.Low"/> <br />
        /// </para> <br />
        /// <para><see langword="{low, high}"/> 的默认值为 <see langword="{10, 1.5}"/></para>
        /// </summary>
        /// <param name="img">图像宽高通道信息</param>
        /// <param name="faceRect">人脸位置信息</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <param name="pointsLength">人脸关键点 数组长度</param>
        /// <param name="level">存储 等级</param>
        /// <param name="score">存储 分数</param>
        /// <param name="low"></param>
        /// <param name="high"></param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "Quality_Integrity", CallingConvention = CallingConvention.Cdecl)]
        public extern static void QualityOfIntegrity(ref FaceImage img, FaceRect faceRect, FaceMarkPoint[] points, int pointsLength, ref int level, ref float score,
            float low = 10f, float high = 1.5f);

        /// <summary>
        /// 姿态评估。
        /// <para>此姿态评估器是传统方式，通过人脸5点坐标值来判断姿态是否为正面。</para>
        /// </summary>
        /// <param name="img">图像宽高通道信息</param>
        /// <param name="faceRect">人脸位置信息</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <param name="pointsLength">人脸关键点 数组长度</param>
        /// <param name="level">存储 等级</param>
        /// <param name="score">存储 分数</param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "Quality_Pose", CallingConvention = CallingConvention.Cdecl)]
        public extern static void QualityOfPose(ref FaceImage img, FaceRect faceRect, FaceMarkPoint[] points, int pointsLength, ref int level, ref float score);

        /// <summary>
        /// 姿态评估 (深度)。
        /// <para>此姿态评估器是深度学习方式，通过回归人头部在yaw、pitch、roll三个方向的偏转角度来评估人脸是否是正面。</para>
        /// <para>
        /// 需要模型 <see langword="pose_estimation.csta"/> 
        /// </para>
        /// </summary>
        /// <param name="img">图像宽高通道信息</param>
        /// <param name="faceRect">人脸位置信息</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <param name="pointsLength">人脸关键点 数组长度</param>
        /// <param name="level">存储 等级</param>
        /// <param name="score">存储 分数</param>
        /// <param name="yawLow">yaw 方向低分数阈值</param>
        /// <param name="yawHigh">yaw 方向高分数阈值</param>
        /// <param name="pitchLow">pitch 方向低分数阈值</param>
        /// <param name="pitchHigh">pitch 方向高分数阈值</param>
        /// <param name="rollLow">roll 方向低分数阈值</param>
        /// <param name="rollHigh">roll 方向高分数阈值</param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "Quality_PoseEx", CallingConvention = CallingConvention.Cdecl)]
        public extern static void QualityOfPoseEx(ref FaceImage img, FaceRect faceRect, FaceMarkPoint[] points, int pointsLength, ref int level, ref float score,
            float yawLow = 25, float yawHigh = 10, float pitchLow = 20, float pitchHigh = 10, float rollLow = 33.33f, float rollHigh = 16.67f);

        /// <summary>
        /// 分辨率评估。
        /// <para>判断人脸部分的分辨率。</para>
        /// <para>
        /// 映射关系为： <br />
        /// • [0, low) => <see cref="QualityLevel.Low"/> <br />
        /// • [low, high) => <see cref="QualityLevel.Medium"/> <br />
        /// • [high, ~) => <see cref="QualityLevel.High"/> <br />
        /// </para> <br />
        /// <para><see langword="{low, high}"/> 的默认值为 <see langword="{80, 120}"/></para>
        /// </summary>
        /// <param name="img">图像宽高通道信息</param>
        /// <param name="faceRect">人脸位置信息</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <param name="pointsLength">人脸关键点 数组长度</param>
        /// <param name="level">存储 等级</param>
        /// <param name="score">存储 分数</param>
        /// <param name="low"></param>
        /// <param name="high"></param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "Quality_Resolution", CallingConvention = CallingConvention.Cdecl)]
        public extern static void QualityOfResolution(ref FaceImage img, FaceRect faceRect, FaceMarkPoint[] points, int pointsLength, ref int level, ref float score,
            float low = 80, float high = 120);

        /// <summary>
        /// 清晰度 (深度)评估。
        /// <para>
        /// 需要模型 <see langword="quality_lbn.csta"/> <br />
        /// 需要模型 <see langword="face_landmarker_pts68.csta"/> 
        /// </para>
        /// <para><see langword="{blur_thresh}"/> 的默认值为 <see langword="{0.8}"/></para>
        /// </summary>
        /// <param name="img">图像宽高通道信息</param>
        /// <param name="faceRect">人脸位置信息</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <param name="pointsLength">人脸关键点 数组长度</param>
        /// <param name="level">存储 等级</param>
        /// <param name="score">存储 分数</param>
        /// <param name="blur_thresh"></param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "Quality_ClarityEx", CallingConvention = CallingConvention.Cdecl)]
        public extern static void QualityOfClarityEx(ref FaceImage img, FaceRect faceRect, FaceMarkPoint[] points, int pointsLength, ref int level, ref float score,
            float blur_thresh = 0.8f);

        /// <summary>
        /// 遮挡评估。
        /// <para>判断人脸部分的分辨率。</para>
        /// </summary>
        /// <param name="img">图像宽高通道信息</param>
        /// <param name="faceRect">人脸位置信息</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <param name="pointsLength">人脸关键点 数组长度</param>
        /// <param name="level">存储 等级</param>
        /// <param name="score">存储 分数</param>
        /// <returns></returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "Quality_NoMask", CallingConvention = CallingConvention.Cdecl)]
        public extern static void QualityOfNoMask(ref FaceImage img, FaceRect faceRect, FaceMarkPoint[] points, int pointsLength, ref int level, ref float score);

        #endregion

        #region 年龄/性别/眼睛状态检测
        #region 年龄预测

        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "GetAgePredictorHandler", CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr GetAgePredictorHandler(int deviceType = 0);

        /// <summary>
        /// 人脸年龄预测
        /// </summary>
        /// <param name="handler">句柄</param>
        /// <param name="img">图像宽高通道信息</param>
        /// <returns>-1 则为失败，否则为预测年龄</returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "PredictAge", CallingConvention = CallingConvention.Cdecl)]
        public extern static int PredictAge(IntPtr handler, ref FaceImage img);

        /// <summary>
        /// 人脸年龄预测（自动裁剪）
        /// </summary>
        /// <param name="handler">句柄</param>
        /// <param name="img">图像宽高通道信息</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <returns>-1 则为失败，否则为预测年龄</returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "PredictAgeWithCrop", CallingConvention = CallingConvention.Cdecl)]
        public extern static int PredictAgeWithCrop(IntPtr handler, ref FaceImage img, FaceMarkPoint[] points);

        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "DisposeAgePredictor", CallingConvention = CallingConvention.Cdecl)]
        public extern static void DisposeAgePredictor(IntPtr handler);

        #endregion

        #region 性别预测

        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "GetGenderPredictorHandler", CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr GetGenderPredictorHandler(int deviceType = 0);

        /// <summary>
        /// 人脸性别预测
        /// </summary>
        /// <param name="handler">句柄</param>
        /// <param name="img">图像宽高通道信息</param>
        /// <returns>-1 则为失败，否则为预测年龄</returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "PredictGender", CallingConvention = CallingConvention.Cdecl)]
        public extern static int PredictGender(IntPtr handler, ref FaceImage img);

        /// <summary>
        /// 人脸性别预测（自动裁剪）
        /// </summary>
        /// <param name="handler">句柄</param>
        /// <param name="img">图像宽高通道信息</param>
        /// <param name="points">人脸关键点 数组</param>
        /// <returns>-1 则为失败，否则为预测年龄</returns>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "PredictGenderWithCrop", CallingConvention = CallingConvention.Cdecl)]
        public extern static int PredictGenderWithCrop(IntPtr handler, ref FaceImage img, FaceMarkPoint[] points);

        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "DisposeGenderPredictor", CallingConvention = CallingConvention.Cdecl)]
        public extern static void DisposeGenderPredictor(IntPtr handler);

        #endregion

        #region 眼睛状态检测

        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "GetEyeStateDetectorHandler", CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr GetEyeStateDetectorHandler(int deviceType = 0);

        /// <summary>
        /// 眼睛状态检测。
        /// </summary>
        /// <param name="handler"></param>
        /// <param name="img">检测的图像数据</param>
        /// <param name="points">人脸关键点</param>
        /// <param name="left_eye"></param>
        /// <param name="right_eye"></param>
        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "EyeStateDetect", CallingConvention = CallingConvention.Cdecl)]
        public extern static void EyeStateDetect(IntPtr handler, ref FaceImage img, FaceMarkPoint[] points, ref int left_eye, ref int right_eye);

        [DllImport(BRIDGE_LIBRARY_NAME, EntryPoint = "DisposeEyeStateDetector", CallingConvention = CallingConvention.Cdecl)]
        public extern static void DisposeEyeStateDetector(IntPtr handler);

        #endregion
        #endregion
    }
}
