using ViewFaceCore.Model;

namespace ViewFaceCore.Configs.QualityConfigs
{
    /// <summary>
    /// 分辨率评估。
    /// <para>判断人脸部分的分辨率。</para>
    /// <para>
    /// 映射关系为： <br />
    /// • [0, low) => <see cref="QualityLevel.Low"/> <br />
    /// • [low, high) => <see cref="QualityLevel.Medium"/> <br />
    /// • [high, ~) => <see cref="QualityLevel.High"/> <br />
    /// </para> <br />
    /// <para><see langword="{low, high}"/> 的默认值为 <see langword="{80, 120}"/></para>
    /// </summary>
    public class ResolutionConfig
    {
        /// <summary>
        /// 默认值为 80
        /// </summary>
        public float Low { get; set; } = 80;
        /// <summary>
        /// 默认值为 120
        /// </summary>
        public float High { get; set; } = 120;
    }
}
