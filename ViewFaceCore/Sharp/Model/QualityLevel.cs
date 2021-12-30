using System.ComponentModel;

namespace ViewFaceCore.Sharp.Model
{
    /// <summary>
    /// 质量评估等级
    /// </summary>
    public enum QualityLevel
    {
        /// <summary>
        /// 错误
        /// </summary>
        [Description("质量评估等级")]
        Error = -1,
        /// <summary>
        /// 质量差
        /// </summary>
        [Description("质量评估等级")]
        Low = 0,
        /// <summary>
        /// 质量一般
        /// </summary>
        [Description("质量评估等级")]
        Medium = 1,
        /// <summary>
        /// 质量高
        /// </summary>
        [Description("质量评估等级")]
        High = 2,
    };
}
