using ViewFaceCore.Configs;

namespace ViewFaceCore.Core;

/// <summary>
/// 预测器
/// </summary>
/// <typeparam name="T"></typeparam>
public abstract class Predictor<T> : BaseViewFace<T> where T : BaseConfig
{
    /// <summary>
    /// 初始化预测器构造器
    /// </summary>
    /// <param name="config"></param>
    public Predictor(T config) : base(config) { }
}
