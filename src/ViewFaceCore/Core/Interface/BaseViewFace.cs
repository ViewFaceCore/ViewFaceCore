using ViewFaceCore.Configs;
using ViewFaceCore.Native;

namespace ViewFaceCore.Core;

/// <summary>
/// 基类
/// </summary>
public abstract class BaseViewFace<T> : IViewFace, IFormattable where T : BaseConfig
{
    /// <summary>
    /// 获取模型路径
    /// </summary>
    public string ModelPath { get => ViewFaceNative.GetModelPath(); }

    /// <summary>
    /// 获取库路径
    /// </summary>
    public string LibraryPath { get => ViewFaceNative.GetLibraryPath(); }

    /// <summary>
    /// 初始化
    /// </summary>
    /// <param name="config"></param>
    public BaseViewFace(T config) => Config = config;

    /// <summary>
    /// 配置
    /// </summary>
    public T Config { get; }

    /// <summary>
    /// 释放标识
    /// </summary>
    protected bool IsDisposed { get; set; } = false;

    /// <summary>
    /// Dispose
    /// </summary>
    public abstract void Dispose();

    #region IFormattable

    /// <summary>
    /// 返回可视化字符串
    /// </summary>
    /// <returns></returns>
    public override string ToString() => ToString(null, null);

    /// <summary>
    /// 返回可视化字符串
    /// </summary>
    /// <param name="format"></param>
    /// <returns></returns>
    public string ToString(string format) => ToString(format, null);

    /// <summary>
    /// 返回可视化字符串
    /// </summary>
    /// <param name="format"></param>
    /// <param name="formatProvider"></param>
    /// <returns></returns>
    public string ToString(string format, IFormatProvider formatProvider)
    {
        string ntips = "CurrentModule", mtips = nameof(ModelPath), otips = "OperatingSystem", atips = "ProcessArchitecture", ltips = "LibraryPath";
        if ((formatProvider ?? Thread.CurrentThread.CurrentCulture) is CultureInfo cultureInfo && cultureInfo.Name.StartsWith("zh"))
        {
            ntips = "当前模块";
            mtips = "模型路径";
            otips = "操作系统";
            atips = "进程架构";
            ltips = "库路径";
        }
        return $"{{{ntips}:{this.GetType().Name}, {otips}:{RuntimeInformation.OSDescription}, {atips}:{RuntimeInformation.ProcessArchitecture}, {mtips}:{ModelPath}, {ltips}:{LibraryPath}}}";
    }

    #endregion
}

