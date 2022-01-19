using System.ComponentModel;

namespace ViewFaceCore.Model
{
    /// <summary>
    /// 质量评估等级
    /// </summary>
    [Description("质量评估等级")]
    public enum QualityLevel
    {
        /// <summary>
        /// 错误
        /// </summary>
        [Description("错误")]
        Error = -1,
        /// <summary>
        /// 质量差
        /// </summary>
        [Description("质量差")]
        Low = 0,
        /// <summary>
        /// 质量一般
        /// </summary>
        [Description("质量一般")]
        Medium = 1,
        /// <summary>
        /// 质量高
        /// </summary>
        [Description("质量高")]
        High = 2,
    };
}
