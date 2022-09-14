namespace ViewFaceCore.Configs;

/// <summary>
/// 姿态评估 (深度) 配置。
/// <para>此姿态评估器是深度学习方式，通过回归人头部在yaw、pitch、roll三个方向的偏转角度来评估人脸是否是正面。</para>
/// </summary>
public sealed class PoseExConfig
{
    /// <summary>
    /// Yaw 方向低分数阈值。
    /// <para>默认值为 25</para>
    /// </summary>
    public float YawLow { get; set; } = 25;
    /// <summary>
    /// Yaw 方向高分数阈值。
    /// <para>默认值为 10</para>
    /// </summary>
    public float YawHigh { get; set; } = 10;
    /// <summary>
    /// Pitch 方向低分数阈值。
    /// <para>默认值为 20</para>
    /// </summary>
    public float PitchLow { get; set; } = 20;
    /// <summary>
    /// Pitch 方向高分数阈值。
    /// <para>默认值为 10</para>
    /// </summary>
    public float PitchHigh { get; set; } = 10;
    /// <summary>
    /// Roll 方向低分数阈值。
    /// <para>默认值为 33.33</para>
    /// </summary>
    public float RollLow { get; set; } = 33.33f;
    /// <summary>
    /// Roll 方向高分数阈值。
    /// <para>默认值为 16.67</para>
    /// </summary>
    public float RollHigh { get; set; } = 16.67f;
}
