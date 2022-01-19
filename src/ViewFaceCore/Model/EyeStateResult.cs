using System;
using System.Globalization;
using System.Threading;
using ViewFaceCore.Extensions;

namespace ViewFaceCore.Model
{
    /// <summary>
    /// 眼睛状态结果
    /// </summary>
    public class EyeStateResult : IFormattable
    {
        /// <summary>
        /// 
        /// </summary>
        public EyeStateResult() { }

        /// <summary>
        /// 使用双眼的状态初始化结果
        /// </summary>
        /// <param name="leftEyeState">左眼状态</param>
        /// <param name="rightEyeState">右眼状态</param>
        public EyeStateResult(EyeState leftEyeState, EyeState rightEyeState)
        {
            LeftEyeState = leftEyeState;
            RightEyeState = rightEyeState;
        }

        /// <summary>
        /// 左眼状态
        /// </summary>
        public EyeState LeftEyeState { get; set; }
        /// <summary>
        /// 右眼状态
        /// </summary>
        public EyeState RightEyeState { get; set; }

        #region IFormattable
        /// <summary>
        /// 返回可视化字符串
        /// </summary>
        /// <returns></returns>
        public override string ToString() => ToString(null, null);
        /// <summary>
        /// 返回可视化字符串
        /// </summary>
        /// <param name="format">D:返回状态的描述文本</param>
        /// <returns></returns>
        public string ToString(string format) => ToString(format, null);
        /// <summary>
        /// 返回可视化字符串
        /// </summary>
        /// <param name="format">D:返回状态的描述文本</param>
        /// <param name="formatProvider"></param>
        /// <returns></returns>
        public string ToString(string format, IFormatProvider formatProvider)
        {
            string ltips = nameof(LeftEyeState), rtips = nameof(RightEyeState);

            if ((formatProvider ?? Thread.CurrentThread.CurrentCulture) is CultureInfo cultureInfo && cultureInfo.Name.StartsWith("zh"))
            { ltips = "左眼状态"; rtips = "右眼状态"; }

            if (format?[0] == 'D')
            { return $"{{{ltips}:{LeftEyeState.ToDescription()}, {rtips}:{RightEyeState.ToDescription()}}}"; }
            else return $"{{{ltips}:{LeftEyeState}, {rtips}:{RightEyeState}}}";
        }
        #endregion
    }
}
