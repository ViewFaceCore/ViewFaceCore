using System.Runtime.InteropServices;
using System.Text;
using ViewFaceCore.Sharp.Model;

namespace ViewFaceCore.Plus
{
    /// <summary>
    /// 日志回调函数
    /// </summary>
    /// <param name="logText"></param>
    public delegate void LogCallBack(string logText);

    /// <summary>
    /// x64 导入方法
    /// </summary>
    class ViewFacePlus64
    {
        const string LibraryPath = @"FaceLibraries\x64\ViewFace.dll";
        /// <summary>
        /// 设置日志回调函数(用于日志打印)
        /// </summary>
        /// <param name="writeLog"></param>
        [DllImport(LibraryPath, EntryPoint = "V_SetLogFunction", CallingConvention = CallingConvention.Cdecl)]
        public static extern void SetLogFunction(LogCallBack writeLog);

        /// <summary>
        /// 设置人脸模型的目录
        /// </summary>
        /// <param name="path"></param>
        [DllImport(LibraryPath, EntryPoint = "V_SetModelPath", CallingConvention = CallingConvention.Cdecl)]
        private extern static void SetModelPath(byte[] path);
        /// <summary>
        /// 设置人脸模型的目录
        /// </summary>
        /// <param name="path"></param>
        public static void SetModelPath(string path) => SetModelPath(Encoding.UTF8.GetBytes(path));

        /// <summary>
        /// 释放使用的资源
        /// </summary>
        [DllImport(LibraryPath, EntryPoint = "V_Dispose", CallingConvention = CallingConvention.Cdecl)]
        public extern static void ViewDispose();

        /// <summary>
        /// 获取人脸模型的目录
        /// </summary>
        /// <param name="path"></param>
        [DllImport(LibraryPath, EntryPoint = "V_GetModelPath", CallingConvention = CallingConvention.Cdecl)]
        private extern static bool GetModelPathEx(ref string path);
        /// <summary>
        /// 获取人脸模型的目录
        /// </summary>
        public static string GetModelPath() { string path = string.Empty; GetModelPathEx(ref path); return path; }

        /// <summary>
        /// 人脸检测器检测到的人脸数量
        /// </summary>
        /// <param name="imgData"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="channels"></param>
        /// <param name="faceSize">最小人脸是人脸检测器常用的一个概念，默认值为20，单位像素。
        /// <para>最小人脸和检测器性能息息相关。主要方面是速度，使用建议上，我们建议在应用范围内，这个值设定的越大越好。SeetaFace采用的是BindingBox Regresion的方式训练的检测器。如果最小人脸参数设置为80的话，从检测能力上，可以将原图缩小的原来的1/4，这样从计算复杂度上，能够比最小人脸设置为20时，提速到16倍。</para>
        /// </param>
        /// <param name="threshold">检测器阈值默认值是0.9，合理范围为[0, 1]。这个值一般不进行调整，除了用来处理一些极端情况。这个值设置的越小，漏检的概率越小，同时误检的概率会提高</param>
        /// <param name="maxWidth">可检测的图像最大宽度。默认值2000。</param>
        /// <param name="maxHeight">可检测的图像最大高度。默认值2000。</param>
        /// <param name="type"></param>
        /// <returns></returns>
        [DllImport(LibraryPath, EntryPoint = "V_DetectorSize", CallingConvention = CallingConvention.Cdecl)]
        public extern static int DetectorSize(byte[] imgData, int width, int height, int channels, double faceSize = 20, double threshold = 0.9, double maxWidth = 2000, double maxHeight = 2000, int type = 0);
        /// <summary>
        /// 人脸检测器
        /// <para>调用此方法前必须先调用 <see cref="DetectorSize(byte[], int, int, int, double, double, double, double, int)"/></para>
        /// </summary>
        /// <param name="score">人脸置信度集合</param>
        /// <param name="x">人脸位置集合</param>
        /// <param name="y">人脸位置集合</param>
        /// <param name="width">人脸大小集合</param>
        /// <param name="height">人脸大小集合</param>
        /// <returns></returns>
        [DllImport(LibraryPath, EntryPoint = "V_Detector", CallingConvention = CallingConvention.Cdecl)]
        public extern static bool Detector(float[] score, int[] x, int[] y, int[] width, int[] height);

        /// <summary>
        /// 人脸关键点数量
        /// </summary>
        /// <returns></returns>
        [DllImport(LibraryPath, EntryPoint = "V_FaceMarkSize", CallingConvention = CallingConvention.Cdecl)]
        public extern static int FaceMarkSize(int type = 0);
        /// <summary>
        /// 人脸关键点
        /// </summary>
        /// <param name="imgData"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="channels"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="fWidth"></param>
        /// <param name="fHeight"></param>
        /// <param name="pointX"></param>
        /// <param name="pointY"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        [DllImport(LibraryPath, EntryPoint = "V_FaceMark", CallingConvention = CallingConvention.Cdecl)]
        public extern static bool FaceMark(byte[] imgData, int width, int height, int channels, int x, int y, int fWidth, int fHeight, double[] pointX, double[] pointY, int type = 0);

        /// <summary>
        /// 提取特征值
        /// </summary>
        /// <param name="imgData"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="channels"></param>
        /// <param name="points"></param>
        /// <param name="features"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        [DllImport(LibraryPath, EntryPoint = "V_Extract", CallingConvention = CallingConvention.Cdecl)]
        public extern static bool Extract(byte[] imgData, int width, int height, int channels, FaceMarkPoint[] points, float[] features, int type = 0);
        /// <summary>
        /// 特征值大小
        /// </summary>
        /// <returns></returns>
        [DllImport(LibraryPath, EntryPoint = "V_ExtractSize", CallingConvention = CallingConvention.Cdecl)]
        public extern static int ExtractSize(int type = 0);

        /// <summary>
        /// 计算相似度
        /// </summary>
        /// <param name="leftFeatures"></param>
        /// <param name="rightFeatures"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        [DllImport(LibraryPath, EntryPoint = "V_CalculateSimilarity", CallingConvention = CallingConvention.Cdecl)]
        public extern static float Similarity(float[] leftFeatures, float[] rightFeatures, int type = 0);
    }

    /// <summary>
    /// x86 导入方法
    /// </summary>
    class ViewFacePlus32
    {
        const string LibraryPath = @"FaceLibraries\x86\ViewFace.dll";
        /// <summary>
        /// 设置日志回调函数(用于日志打印)
        /// </summary>
        /// <param name="writeLog"></param>
        [DllImport(LibraryPath, EntryPoint = "V_SetLogFunction", CallingConvention = CallingConvention.Cdecl)]
        public static extern void SetLogFunction(LogCallBack writeLog);

        /// <summary>
        /// 设置人脸模型的目录
        /// </summary>
        /// <param name="path"></param>
        [DllImport(LibraryPath, EntryPoint = "V_SetModelPath", CallingConvention = CallingConvention.Cdecl)]
        private extern static void SetModelPath(byte[] path);
        /// <summary>
        /// 设置人脸模型的目录
        /// </summary>
        /// <param name="path"></param>
        public static void SetModelPath(string path) => SetModelPath(Encoding.UTF8.GetBytes(path));

        /// <summary>
        /// 释放使用的资源
        /// </summary>
        [DllImport(LibraryPath, EntryPoint = "V_Dispose", CallingConvention = CallingConvention.Cdecl)]
        public extern static void ViewDispose();

        /// <summary>
        /// 获取人脸模型的目录
        /// </summary>
        /// <param name="path"></param>
        [DllImport(LibraryPath, EntryPoint = "V_GetModelPath", CallingConvention = CallingConvention.Cdecl)]
        private extern static bool GetModelPathEx(ref string path);
        /// <summary>
        /// 获取人脸模型的目录
        /// </summary>
        public static string GetModelPath() { string path = string.Empty; GetModelPathEx(ref path); return path; }

        /// <summary>
        /// 人脸检测器检测到的人脸数量
        /// </summary>
        /// <param name="imgData"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="channels"></param>
        /// <param name="faceSize">最小人脸是人脸检测器常用的一个概念，默认值为20，单位像素。
        /// <para>最小人脸和检测器性能息息相关。主要方面是速度，使用建议上，我们建议在应用范围内，这个值设定的越大越好。SeetaFace采用的是BindingBox Regresion的方式训练的检测器。如果最小人脸参数设置为80的话，从检测能力上，可以将原图缩小的原来的1/4，这样从计算复杂度上，能够比最小人脸设置为20时，提速到16倍。</para>
        /// </param>
        /// <param name="threshold">检测器阈值默认值是0.9，合理范围为[0, 1]。这个值一般不进行调整，除了用来处理一些极端情况。这个值设置的越小，漏检的概率越小，同时误检的概率会提高</param>
        /// <param name="maxWidth">可检测的图像最大宽度。默认值2000。</param>
        /// <param name="maxHeight">可检测的图像最大高度。默认值2000。</param>
        /// <param name="type"></param>
        /// <returns></returns>
        [DllImport(LibraryPath, EntryPoint = "V_DetectorSize", CallingConvention = CallingConvention.Cdecl)]
        public extern static int DetectorSize(byte[] imgData, int width, int height, int channels, double faceSize = 20, double threshold = 0.9, double maxWidth = 2000, double maxHeight = 2000, int type = 0);
        /// <summary>
        /// 人脸检测器
        /// <para>调用此方法前必须先调用 <see cref="DetectorSize(byte[], int, int, int, double, double, double, double, int)"/></para>
        /// </summary>
        /// <param name="score">人脸置信度集合</param>
        /// <param name="x">人脸位置集合</param>
        /// <param name="y">人脸位置集合</param>
        /// <param name="width">人脸大小集合</param>
        /// <param name="height">人脸大小集合</param>
        /// <returns></returns>
        [DllImport(LibraryPath, EntryPoint = "V_Detector", CallingConvention = CallingConvention.Cdecl)]
        public extern static bool Detector(float[] score, int[] x, int[] y, int[] width, int[] height);

        /// <summary>
        /// 人脸关键点数量
        /// </summary>
        /// <returns></returns>
        [DllImport(LibraryPath, EntryPoint = "V_FaceMarkSize", CallingConvention = CallingConvention.Cdecl)]
        public extern static int FaceMarkSize(int type = 0);
        /// <summary>
        /// 人脸关键点
        /// </summary>
        /// <param name="imgData"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="channels"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="fWidth"></param>
        /// <param name="fHeight"></param>
        /// <param name="pointX"></param>
        /// <param name="pointY"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        [DllImport(LibraryPath, EntryPoint = "V_FaceMark", CallingConvention = CallingConvention.Cdecl)]
        public extern static bool FaceMark(byte[] imgData, int width, int height, int channels, int x, int y, int fWidth, int fHeight, double[] pointX, double[] pointY, int type = 0);

        /// <summary>
        /// 提取特征值
        /// </summary>
        /// <param name="imgData"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="channels"></param>
        /// <param name="points"></param>
        /// <param name="features"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        [DllImport(LibraryPath, EntryPoint = "V_Extract", CallingConvention = CallingConvention.Cdecl)]
        public extern static bool Extract(byte[] imgData, int width, int height, int channels, FaceMarkPoint[] points, float[] features, int type = 0);
        /// <summary>
        /// 特征值大小
        /// </summary>
        /// <returns></returns>
        [DllImport(LibraryPath, EntryPoint = "V_ExtractSize", CallingConvention = CallingConvention.Cdecl)]
        public extern static int ExtractSize(int type = 0);

        /// <summary>
        /// 计算相似度
        /// </summary>
        /// <param name="leftFeatures"></param>
        /// <param name="rightFeatures"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        [DllImport(LibraryPath, EntryPoint = "V_CalculateSimilarity", CallingConvention = CallingConvention.Cdecl)]
        public extern static float Similarity(float[] leftFeatures, float[] rightFeatures, int type = 0);
    }
}
