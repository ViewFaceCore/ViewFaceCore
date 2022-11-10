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
    /// 默认只支持CPU，GPU需要自行编译或引入已经编译好的GPU版本
    /// </remarks>
    public DeviceType DeviceType { get; set; } = DeviceType.AUTO;
}
