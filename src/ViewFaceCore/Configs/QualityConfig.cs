namespace ViewFaceCore.Configs;

/// <summary>
/// 质量评估配置。
/// </summary>
public class QualityConfig : BaseConfig
{
    private BrightnessConfig brightness = new BrightnessConfig();

    /// <summary>
    /// 获取或设置亮度评估配置。
    /// </summary>
    public BrightnessConfig Brightness
    {
        get
        {
            return brightness;
        }
        set
        {
            if (value == null)
            {
                throw new ArgumentNullException(nameof(Brightness));
            }
            brightness = value;
        }
    }

    private ClarityConfig clarity = new ClarityConfig();

    /// <summary>
    /// 获取或设置清晰度评估配置。
    /// </summary>
    public ClarityConfig Clarity
    {
        get
        {
            return clarity;
        }
        set
        {
            if (value == null)
            {
                throw new ArgumentNullException(nameof(Clarity));
            }
            clarity = value;
        }
    }

    private IntegrityConfig integrity = new IntegrityConfig();

    /// <summary>
    /// 获取或设置完整度评估配置。
    /// </summary>
    public IntegrityConfig Integrity
    {
        get
        {
            return integrity;
        }
        set
        {
            if (value == null)
            {
                throw new ArgumentNullException(nameof(Integrity));
            }
            integrity = value;
        }
    }

    private PoseExConfig poseEx = new PoseExConfig();

    /// <summary>
    /// 获取或设置姿态评估 (深度)配置。
    /// </summary>
    public PoseExConfig PoseEx
    {
        get
        {
            return poseEx;
        }
        set
        {
            if (value == null)
            {
                throw new ArgumentNullException(nameof(PoseEx));
            }
            poseEx = value;
        }
    }

    private ResolutionConfig resolution = new ResolutionConfig();

    /// <summary>
    /// 获取或设置分辨率评估配置。
    /// </summary>
    public ResolutionConfig Resolution
    {
        get
        {
            return resolution;
        }
        set
        {
            if (value == null)
            {
                throw new ArgumentNullException(nameof(Resolution));
            }
            resolution = value;
        }
    }

    private ClarityExConfig clarityEx = new ClarityExConfig();

    /// <summary>
    /// 获取或设置清晰度 (深度)评估配置。
    /// </summary>
    public ClarityExConfig ClarityEx
    {
        get
        {
            return clarityEx;
        }
        set
        {
            if(value == null)
            {
                throw new ArgumentNullException(nameof(ClarityEx));
            }
            clarityEx = value;
        }
    }
}
