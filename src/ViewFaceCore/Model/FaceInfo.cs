using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;

namespace ViewFaceCore.Model
{
    /// <summary>
    /// 面部信息
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct FaceInfo : IFormattable
    {
        private readonly FaceRect pos;
        private readonly float score;

        /// <summary>
        /// 人脸置信度
        /// </summary>
        public float Score => score;
        /// <summary>
        /// 人脸位置
        /// </summary>
        public FaceRect Location => pos;

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
            string stips = nameof(Score), ltips = nameof(Location);

            if ((formatProvider ?? Thread.CurrentThread.CurrentCulture) is CultureInfo cultureInfo && cultureInfo.Name.StartsWith("zh"))
            { stips = "置信度"; ltips = "位置"; }

            return $"{{{stips}:{Score}, {ltips}:{Location.ToString(format, formatProvider)}}}";
        }
        #endregion
    }
}
