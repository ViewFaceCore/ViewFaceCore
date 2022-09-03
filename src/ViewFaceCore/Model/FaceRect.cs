using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;

namespace ViewFaceCore.Model
{
    /// <summary>
    /// 人脸位置矩形
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct FaceRect : IFormattable
    {
        private readonly int x;
        private readonly int y;
        private readonly int width;
        private readonly int height;

        /// <summary>
        /// 左上角点横坐标
        /// </summary>
        public int X => x;

        /// <summary>
        /// 左上角点纵坐标
        /// </summary>
        public int Y => y;

        /// <summary>
        /// 矩形宽度
        /// </summary>
        public int Width => width;

        /// <summary>
        /// 矩形高度
        /// </summary>
        public int Height => height;

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
            string xtips = nameof(X), ytips = nameof(Y), wtips = nameof(Width), htips = nameof(Height);

            if ((formatProvider ?? Thread.CurrentThread.CurrentCulture) is CultureInfo cultureInfo && cultureInfo.Name.StartsWith("zh"))
            { xtips = "X坐标"; ytips = "Y坐标"; wtips = "宽度"; htips = "高度"; }

            return $"{{{xtips}:{X}, {ytips}:{Y}, {wtips}:{Width}, {htips}:{Height}}}";
        }
        #endregion
    }
}
