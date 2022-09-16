namespace ViewFaceCore.Models;

/// <summary>
/// 活体检测结果
/// </summary>
public sealed class AntiSpoofingResult
{
    internal AntiSpoofingResult(AntiSpoofingStatus status, float clarity, float reality)
    {
        this.Status = status;
        this.Clarity = clarity;
        this.Reality = reality;
    }

    /// <summary>
    /// 活体检测状态
    /// </summary>
    public AntiSpoofingStatus Status { get; }

    /// <summary>
    /// 清晰度
    /// </summary>
    public float Clarity { get; }

    /// <summary>
    /// 真实度
    /// </summary>
    public float Reality { get; }
}

/// <summary>
/// 活体检测状态
/// </summary>
[Description("活体检测状态")]
public enum AntiSpoofingStatus
{
    /// <summary>
    /// 错误或没有找到指定的人脸索引处的人脸
    /// </summary>
    [Description("错误或没有找到指定的人脸索引处的人脸")]
    Error = -1,
    /// <summary>
    /// 真实人脸
    /// </summary>
    [Description("真实人脸")]
    Real = 0,
    /// <summary>
    /// 攻击人脸（假人脸）
    /// </summary>
    [Description("攻击人脸（假人脸）")]
    Spoof = 1,
    /// <summary>
    /// 无法判断（人脸成像质量不好）
    /// </summary>
    [Description("无法判断（人脸成像质量不好）")]
    Fuzzy = 2,
    /// <summary>
    /// 正在检测
    /// </summary>
    [Description("正在检测")]
    Detecting = 3,
};
