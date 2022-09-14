using ViewFaceCore.Configs.Enums;

namespace ViewFaceCore.Configs;

/// <summary>
/// 通用配置
/// </summary>
public abstract class BaseConfig
{
    /// <summary>
    /// 识别用的设备类型
    /// </summary>
    /// <remarks>
    /// 目前只能作用CPU，GPU无法使用
    /// </remarks>
    public DeviceType DeviceType { get; set; } = DeviceType.AUTO;
}
