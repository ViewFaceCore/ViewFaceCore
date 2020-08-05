using ViewFaceCore.Sharp.Model;

namespace ViewFaceCore.Sharp.Configs.QualityConfigs
{
    /// <summary>
    /// 完整度评估器配置。
    /// <para>完整度评估是朴素的判断人来是否因为未完全进入摄像头而造成的不完整的情况。该方法不适用于判断遮挡造成的不完整。</para>
    /// <para>
    /// 映射关系为： <br />
    /// • 人脸外扩 high 倍数没有超出图像 => <see cref="QualityLevel.High"/> <br />
    /// • 人脸外扩 low 倍数没有超出图像 => <see cref="QualityLevel.Medium"/> <br />
    /// • 其他 => <see cref="QualityLevel.Low"/> <br />
    /// </para> <br />
    /// <para><see langword="{low, high}"/> 的默认值为 <see langword="{10, 1.5}"/></para>
    /// </summary>
    public class IntegrityConfig
    {
        /// <summary>
        /// 默认值为 10
        /// </summary>
        public float Low { get; set; } = 10;
        /// <summary>
        /// 默认值为 1.5
        /// </summary>
        public float High { get; set; } = 1.5f;
    }
}
