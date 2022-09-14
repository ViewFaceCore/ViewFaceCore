using System.ComponentModel;

namespace ViewFaceCore.Configs.Enums;

/// <summary>
/// 人脸类型
/// </summary>
public enum FaceType : int
{
    /// <summary>
    /// 高精度人脸识别模型
    /// </summary>
    [Description("高精度人脸识别模型")]
    Normal = 0,

    /// <summary>
    /// 戴口罩人脸识别模型
    /// </summary>
    [Description("戴口罩人脸识别模型")]
    Mask,

    /// <summary>
    /// 轻量级人脸识别模型
    /// </summary>
    [Description("轻量级人脸识别模型")]
    Light,
}
