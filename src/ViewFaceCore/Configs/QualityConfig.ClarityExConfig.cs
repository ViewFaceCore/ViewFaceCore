namespace ViewFaceCore.Configs;

/// <summary>
/// 清晰度 (深度)评估。
/// <para>
/// 需要模型 <see langword="quality_lbn.csta"/> <br />
/// 需要模型 <see langword="face_landmarker_pts68.csta"/> 
/// </para>
/// <para><see langword="{blur_thresh}"/> 的默认值为 <see langword="{0.8}"/></para>
/// </summary>
public sealed class ClarityExConfig
{
    /// <summary>
    /// 评估对应分值超过选项之后就认为是模糊图片
    /// </summary>
    public float BlurThresh { get; set; } = 0.8f;
}
