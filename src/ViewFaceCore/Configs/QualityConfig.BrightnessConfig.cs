using ViewFaceCore.Configs.Enums;

namespace ViewFaceCore.Configs;

/// <summary>
/// 亮度评估器配置。
/// <para>
/// 评估器会将综合的亮度从灰度值映射到 <see cref="QualityLevel"/>，其映射关系为： <br />
/// • [0, v0), [v3, ~) => <see cref="QualityLevel.Low"/> <br />
/// • [v0, v1), [v2, v3) => <see cref="QualityLevel.Medium"/> <br />
/// • [v1, v2) => <see cref="QualityLevel.High"/> <br />
/// </para>
/// </summary>
public sealed class BrightnessConfig
{
    /// <summary>
    /// 默认值为 70
    /// </summary>
    public byte V0 { get; set; } = 70;
    /// <summary>
    /// 默认值为 100
    /// </summary>
    public byte V1 { get; set; } = 100;
    /// <summary>
    /// 默认值为 210
    /// </summary>
    public byte V2 { get; set; } = 210;
    /// <summary>
    /// 默认值为 230
    /// </summary>
    public byte V3 { get; set; } = 230;
}
