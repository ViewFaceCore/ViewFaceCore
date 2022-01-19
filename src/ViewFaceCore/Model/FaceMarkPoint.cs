using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;

namespace ViewFaceCore.Model
{
    /// <summary>
    /// 点坐标
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct FaceMarkPoint : IFormattable
    {
        private readonly double x;
        private readonly double y;

        /// <summary>
        /// 横坐标
        /// </summary>
        public double X => x;
        /// <summary>
        /// 纵坐标
        /// </summary>
        public double Y => y;

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
            string xtips = nameof(X), ytips = nameof(Y);

            if ((formatProvider ?? Thread.CurrentThread.CurrentCulture) is CultureInfo cultureInfo && cultureInfo.Name.StartsWith("zh"))
            { xtips = "X坐标"; ytips = "Y坐标"; }

            return $"{{{xtips}:{X}, {ytips}:{Y}}}";
        }
        #endregion
    }
}
