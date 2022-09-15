namespace ViewFaceCore.Models;

/// <summary>
/// 戴口罩识别返回模型
/// </summary>
public sealed class PlotMaskResult
{
    internal PlotMaskResult(float score, bool status, bool masked)
    {
        Score = score;
        Status = status;
        Masked = masked;
    }

    /// <summary>
    /// 评估分数
    /// </summary>
    public float Score { get; }

    /// <summary>
    /// 是否检测成功
    /// </summary>
    public bool Status { get; }

    /// <summary>
    /// 是否戴口罩
    /// </summary>
    public bool Masked { get; }
}
