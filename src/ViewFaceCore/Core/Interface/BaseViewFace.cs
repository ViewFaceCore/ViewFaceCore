using System;
using ViewFaceCore.Native;

namespace ViewFaceCore.Core
{
    /// <summary>
    /// 基类
    /// </summary>
    public abstract class BaseViewFace : IFormattable
    {
        /// <summary>
        /// 获取或设置模型路径
        /// </summary>
        public string ModelPath { get => ViewFaceNative.GetModelPath(); }

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
            //string mtips = nameof(ModelPath), otips = "OperatingSystem", atips = "ProcessArchitecture";

            //if ((formatProvider ?? Thread.CurrentThread.CurrentCulture) is CultureInfo cultureInfo && cultureInfo.Name.StartsWith("zh"))
            //{ mtips = "模型路径"; otips = "操作系统"; atips = "进程架构"; }

            //return $"{{{mtips}:{ModelPath}, {otips}:{RuntimeInformation.OSDescription}, {atips}:{RuntimeInformation.ProcessArchitecture}}}";

            return "";
        }

        #endregion
    }

}
