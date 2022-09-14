namespace ViewFaceCore.Configs;

/// <summary>
/// 口罩检测配置
/// </summary>
public class MaskDetectConfig : BaseConfig
{
    /// <summary>
    /// 阈值，默认0.5
    /// </summary>
    /// <remarks>
    /// 一般性的，score超过0.5，则认为是检测带上了口罩
    /// </remarks>
    public float Threshold { get; set; } = 0.5f;
}
