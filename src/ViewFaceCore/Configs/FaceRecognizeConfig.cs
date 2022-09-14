using System.Collections.Generic;
using ViewFaceCore.Configs.Enums;

namespace ViewFaceCore.Configs;

/// <summary>
/// 人脸对比阈值配置
/// </summary>
public sealed class FaceRecognizeConfig : BaseConfig
{
    /// <summary>
    /// 人类类型
    /// </summary>
    public FaceType FaceType { get; set; } = FaceType.Normal;

    /// <summary>
    /// 人脸相似度阈值
    /// </summary>
    private Dictionary<FaceType, float> thresholds = new Dictionary<FaceType, float>()
    {
        { FaceType.Normal, 0.62f },
        { FaceType.Mask, 0.48f },
        { FaceType.Light, 0.55f },
    };

    /// <summary>
    /// 获取默认人脸识别模型的相似度阈值。
    /// </summary>
    /// <param name="type"></param>
    /// <returns></returns>
    public float Threshold => thresholds[FaceType];

    /// <summary>
    /// 设置指定人脸识别模型的相似度阈值。
    /// <para>
    /// 默认阈值是一般场景使用的推荐阈值。一般来说1比1的场景下，该阈值会对应偏低，1比N场景会对应偏高。
    /// </para>
    /// <para>
    /// • <see cref="FaceType.Normal"/> 的默认相似度阈值为 <see langword="0.62"/> <br />
    /// • <see cref="FaceType.Mask"/> 的默认相似度阈值为 <see langword="0.48"/> <br />
    /// • <see cref="FaceType.Light"/> 的默认相似度阈值为 <see langword="0.55"/> <br />
    /// </para>
    /// </summary>
    /// <param name="type">指定的人脸识别模型</param>
    /// <param name="score">相似度阈值</param>
    public void SetThreshold(FaceType type, float score) => thresholds[type] = score;
}
