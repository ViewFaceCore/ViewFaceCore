using System.ComponentModel;

namespace ViewFaceCore.Model
{
    /// <summary>
    /// 眼睛状态
    /// </summary>
    [Description("眼睛状态")]
    public enum EyeState
    {
        /// <summary>
        /// 眼睛闭合
        /// </summary>
        [Description("闭眼")]
        Close,
        /// <summary>
        /// 眼睛张开
        /// </summary>
        [Description("睁眼")]
        Open,
        /// <summary>
        /// 不是眼睛区域
        /// </summary>
        [Description("不是眼睛区域")]
        Random,
        /// <summary>
        /// 状态无法判断
        /// </summary>
        [Description("无法判断")]
        Unknown
    }
}
