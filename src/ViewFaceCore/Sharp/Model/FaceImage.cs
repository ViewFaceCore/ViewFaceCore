using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

#if !NET6_0_OR_GREATER
using System.Drawing;
using ViewFaceCore.Extension;
#endif

namespace ViewFaceCore.Sharp.Model
{
    /// <summary>
    /// 人脸图像信息数据
    /// <para>图像 data 直接使用 byte[] 传送，在 ViewFace 内部进行组装</para>
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct FaceImage : IDisposable
    {
        private readonly int width;
        private readonly int height;
        private readonly int channels;
        private readonly IntPtr data;

        /// <summary>
        /// 构造器
        /// </summary>
        /// <param name="width">人脸图像宽度</param>
        /// <param name="height">人脸图像高度</param>
        /// <param name="channels">人脸图像通道数</param>
        /// <param name="buffer">人脸图像像素数据<para>按照 BGR 排列的 <see cref="byte"/> 列表</para></param>
        public FaceImage(int width, int height, int channels, byte[] buffer)
        {
            this.width = width;
            this.height = height;
            this.channels = channels;

            int size = Marshal.SizeOf(typeof(byte));
            var data = Marshal.AllocHGlobal(size * width * height * channels);

            int index = 0;
            foreach (var @byte in buffer)
            {
                Marshal.WriteByte(data, index * size, @byte);
                index++;
            }
            if (index != width * height * channels)
            { throw new ArgumentOutOfRangeException(nameof(buffer), "data.Length ≠ width × height × channels"); }
            this.data = data;
        }

        /// <summary>
        /// 获取人脸图像宽度
        /// </summary>
        public int Width => width;
        /// <summary>
        /// 获取人脸图像高度
        /// </summary>
        public int Height => height;
        /// <summary>
        /// 获取人脸图像通道数
        /// </summary>
        public int Channels => channels;
        /// <summary>
        /// 获取人脸图像BGR数据
        /// </summary>
        public IEnumerable<byte> Data
        {
            get
            {
                int length = width * height * channels;
                int size = Marshal.SizeOf(typeof(byte));
                for (int i = 0; i < length; i++)
                {
                    yield return Marshal.ReadByte(data, i * size);
                }
            }
        }

        /// <summary>
        /// <see cref="IDisposable"/>
        /// </summary>
        public void Dispose()
        {
            Marshal.FreeHGlobal(data);
        }
    };
}
