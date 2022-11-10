using ViewFaceCore.Configs.Enums;

namespace ViewFaceCore.Configs;

/// <summary>
/// 人脸对比阈值配置
/// </summary>
public sealed class FaceRecognizeConfig : BaseConfig
{
    /// <summary>
    /// 人脸相似度阈值
    /// </summary>
    private readonly Dictionary<FaceType, float> _thresholds = new Dictionary<FaceType, float>()
    {
        { FaceType.Normal, 0.62f },
        { FaceType.Mask, 0.48f },
        { FaceType.Light, 0.55f },
    };

    /// <summary>
    /// 人脸对比阈值配置
    /// <para>
    /// 默认阈值是一般场景使用的推荐阈值。一般来说1比1的场景下，该阈值会对应偏低，1比N场景会对应偏高。
    /// </para>
    /// <para>
    /// • <see cref="FaceType.Normal"/> 的默认相似度阈值为 <see langword="0.62"/> <br />
    /// • <see cref="FaceType.Mask"/> 的默认相似度阈值为 <see langword="0.48"/> <br />
    /// • <see cref="FaceType.Light"/> 的默认相似度阈值为 <see langword="0.55"/> <br />
    /// </para>
    /// </summary>
    public FaceRecognizeConfig(FaceType faceType = FaceType.Normal)
    {
        this.FaceType = faceType;
    }

    /// <summary>
    /// 人脸对比阈值配置
    /// <para>
    /// 默认阈值是一般场景使用的推荐阈值。一般来说1比1的场景下，该阈值会对应偏低，1比N场景会对应偏高。
    /// </para>
    /// <para>
    /// • <see cref="FaceType.Normal"/> 的默认相似度阈值为 <see langword="0.62"/> <br />
    /// • <see cref="FaceType.Mask"/> 的默认相似度阈值为 <see langword="0.48"/> <br />
    /// • <see cref="FaceType.Light"/> 的默认相似度阈值为 <see langword="0.55"/> <br />
    /// </para>
    /// </summary>
    public FaceRecognizeConfig(FaceType faceType, float threshold)
    {
        this.FaceType = faceType;
        this.Threshold = threshold;
    }

    /// <summary>
    /// 人脸类型
    /// </summary>
    public FaceType FaceType { get; private set; } = FaceType.Normal;

    /// <summary>
    /// 获取或者设置默认人脸识别模型的相似度阈值。
    /// </summary>
    /// <returns></returns>
    public float Threshold
    {
        get
        {
            return _thresholds[this.FaceType];
        }
        set
        {
            if(_thresholds.ContainsKey(this.FaceType))
            _thresholds[this.FaceType] = value;
        }
    }
}
