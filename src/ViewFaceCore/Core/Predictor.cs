using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using ViewFaceCore.Configs;
using ViewFaceCore.Native;

namespace ViewFaceCore.Core;

/// <summary>
/// 预测器
/// </summary>
public interface IPredictor : IDisposable
{

}

/// <summary>
/// 预测器
/// </summary>
/// <typeparam name="T"></typeparam>
public abstract class Predictor<T> : IPredictor where T : BaseConfig
{
    /// <summary>
    /// 初始化
    /// </summary>
    /// <param name="config"></param>
    public Predictor(T config) => Config = config;

    /// <summary>
    /// 配置
    /// </summary>
    public T Config { get; }

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
        string ntips = "CurrentModule", otips = "OperatingSystem", atips = "ProcessArchitecture";
        if ((formatProvider ?? Thread.CurrentThread.CurrentCulture) is CultureInfo cultureInfo && cultureInfo.Name.StartsWith("zh"))
        {
            ntips = "当前模块";
            otips = "操作系统";
            atips = "进程架构";
        }
        return $"{{{ntips}:{this.GetType().Name}, {otips}:{RuntimeInformation.OSDescription}, {atips}:{RuntimeInformation.ProcessArchitecture}}}";
    }

    #endregion

    /// <inheritdoc/>
    public abstract void Dispose();
}
