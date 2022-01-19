using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;

namespace ViewFaceCore.Model
{
    /// <summary>
    /// 人脸图像信息数据
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct FaceImage : IDisposable, IEquatable<FaceImage>, IFormattable
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

#pragma warning disable CS1591 // 缺少对公共可见类型或成员的 XML 注释
        public static bool operator ==(FaceImage a, FaceImage b)
            => a.Width == b.Width
            && a.Height == b.Height
            && a.Channels == b.Channels
            && EqualityComparer<IEnumerable<byte>>.Default.Equals(a.Data, b.Data);

        public static bool operator !=(FaceImage a, FaceImage b)
            => a.Width != b.Width
            || a.Height != b.Height
            || a.Channels != b.Channels
            || !EqualityComparer<IEnumerable<byte>>.Default.Equals(a.Data, b.Data);

        public override bool Equals(object obj)
        {
            if (obj is FaceImage other)
            { return this == other; }
            else { return false; }
        }

        public override int GetHashCode()
        {
            int hashCode = 1213452459;
            hashCode = hashCode * -1521134295 + Width.GetHashCode();
            hashCode = hashCode * -1521134295 + Height.GetHashCode();
            hashCode = hashCode * -1521134295 + Channels.GetHashCode();
            hashCode = hashCode * -1521134295 + EqualityComparer<IEnumerable<byte>>.Default.GetHashCode(Data);
            return hashCode;
        }
        public bool Equals(FaceImage other) => this == other;

        public void Dispose()
        {
            Marshal.FreeHGlobal(data);
        }
#pragma warning restore CS1591 // 缺少对公共可见类型或成员的 XML 注释


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
            string wtips = nameof(Width), htips = nameof(Height), ctips = nameof(Channels);

            if ((formatProvider ?? Thread.CurrentThread.CurrentCulture) is CultureInfo cultureInfo && cultureInfo.Name.StartsWith("zh"))
            { wtips = "宽度"; htips = "高度"; ctips = "通道数"; }

            return $"{{{wtips}:{Width}, {htips}:{Height}, {ctips}:{Channels}}}";
        }
        #endregion
    }
}
