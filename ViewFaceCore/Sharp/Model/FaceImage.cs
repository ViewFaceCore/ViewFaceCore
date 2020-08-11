using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace ViewFaceCore.Sharp.Model
{
    /// <summary>
    /// 人脸图像信息数据
    /// <para>图像 data 直接使用 byte[] 传送，在 ViewFace 内部进行组装</para>
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct FaceImage
    {
        /// <summary>
        /// 构造器
        /// </summary>
        /// <param name="width">人脸图像宽度</param>
        /// <param name="height">人脸图像高度</param>
        /// <param name="channels">人脸图像通道数</param>
        public FaceImage(int width, int height, int channels)
        {
            this.width = width;
            this.height = height;
            this.width = width;
            this.channels = channels;
            this.data = null;
        }

        private int width;
        private int height;
        private int channels;
        [MarshalAs(UnmanagedType.ByValArray, ArraySubType = UnmanagedType.U1)]
        private byte[] data;

        /// <summary>
        /// 获取或设置宽度
        /// </summary>
        public int Width { get => width; set => width = value; }
        /// <summary>
        /// 获取或设置高度
        /// </summary>
        public int Height { get => height; set => height = value; }
        /// <summary>
        /// 获取或设置通道数
        /// </summary>
        public int Channels { get => channels; set => channels = value; }
    };
}
