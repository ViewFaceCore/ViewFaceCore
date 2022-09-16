namespace ViewFaceCore.Configs.Enums;

/// <summary>
/// 关键点类型
/// </summary>
[Description("关键点类型")]
public enum MarkType : int
{
    /// <summary>
    /// 68个关键点
    /// </summary>
    [Description("68个关键点")]
    Normal = 0,

    /// <summary>
    /// 戴口罩的关键点
    /// </summary>
    [Description("戴口罩的关键点")]
    Mask,

    /// <summary>
    /// 5个关键点
    /// </summary>
    [Description("5个关键点")]
    Light,
}