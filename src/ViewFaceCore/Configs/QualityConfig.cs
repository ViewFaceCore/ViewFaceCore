namespace ViewFaceCore.Configs;

/// <summary>
/// 质量评估配置。
/// </summary>
public class QualityConfig : BaseConfig
{
    /// <summary>
    /// 获取或设置亮度评估配置。
    /// </summary>
    public BrightnessConfig Brightness { get; set; } = new BrightnessConfig();

    /// <summary>
    /// 获取或设置清晰度评估配置。
    /// </summary>
    public ClarityConfig Clarity { get; set; } = new ClarityConfig();

    /// <summary>
    /// 获取或设置完整度评估配置。
    /// </summary>
    public IntegrityConfig Integrity { get; set; } = new IntegrityConfig();

    /// <summary>
    /// 获取或设置姿态评估 (深度)配置。
    /// </summary>
    public PoseExConfig PoseEx { get; set; } = new PoseExConfig();

    /// <summary>
    /// 获取或设置分辨率评估配置。
    /// </summary>
    public ResolutionConfig Resolution { get; set; } = new ResolutionConfig();

    /// <summary>
    /// 获取或设置清晰度 (深度)评估配置。
    /// </summary>
    public ClarityExConfig ClarityEx { get; set; } = new ClarityExConfig();
}
