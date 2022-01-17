using ViewFaceCore.Model;

namespace ViewFaceCore.Configs.QualityConfigs
{
    /// <summary>
    /// 清晰度评估器配置。
    /// <para>清晰度这里是传统方式通过二次模糊后图像信息损失程度统计的清晰度。</para>
    /// <para>
    /// 映射关系为： <br />
    /// • [0, low) => <see cref="QualityLevel.Low"/> <br />
    /// • [low, high) => <see cref="QualityLevel.Medium"/> <br />
    /// • [high, ~) => <see cref="QualityLevel.High"/> <br />
    /// </para> <br />
    /// <para><see langword="{low, high}"/> 的默认值为 <see langword="{0.1, 0.2}"/></para>
    /// </summary>
    public class ClarityConfig
    {
        /// <summary>
        /// 默认值为 0.1
        /// </summary>
        public float Low { get; set; } = 0.1f;
        /// <summary>
        /// 默认值为 0.2
        /// </summary>
        public float High { get; set; } = 0.2f;
    }
}
