using ViewFaceCore.Extensions;

namespace ViewFaceCore.Models;

/// <summary>
/// 眼睛状态结果
/// </summary>
public sealed class EyeStateResult : IFormattable
{
    /// <summary>
    /// 使用双眼的状态初始化结果
    /// </summary>
    /// <param name="leftEyeState">左眼状态</param>
    /// <param name="rightEyeState">右眼状态</param>
    internal EyeStateResult(EyeState leftEyeState, EyeState rightEyeState)
    {
        LeftEyeState = leftEyeState;
        RightEyeState = rightEyeState;
    }

    /// <summary>
    /// 左眼状态
    /// </summary>
    public EyeState LeftEyeState { get; }
    /// <summary>
    /// 右眼状态
    /// </summary>
    public EyeState RightEyeState { get; }

    #region IFormattable
    /// <summary>
    /// 返回可视化字符串
    /// </summary>
    /// <returns></returns>
    public override string ToString() => ToString(null, null);
    /// <summary>
    /// 返回可视化字符串
    /// </summary>
    /// <param name="format">D:返回状态的描述文本</param>
    /// <returns></returns>
    public string ToString(string format) => ToString(format, null);
    /// <summary>
    /// 返回可视化字符串
    /// </summary>
    /// <param name="format">D:返回状态的描述文本</param>
    /// <param name="formatProvider"></param>
    /// <returns></returns>
    public string ToString(string format, IFormatProvider formatProvider)
    {
        string ltips = nameof(LeftEyeState), rtips = nameof(RightEyeState);

        if ((formatProvider ?? Thread.CurrentThread.CurrentCulture) is CultureInfo cultureInfo && cultureInfo.Name.StartsWith("zh"))
        { ltips = "左眼状态"; rtips = "右眼状态"; }

        if (format?[0] == 'D')
        { return $"{{{ltips}:{LeftEyeState.ToDescription()}, {rtips}:{RightEyeState.ToDescription()}}}"; }
        else return $"{{{ltips}:{LeftEyeState}, {rtips}:{RightEyeState}}}";
    }
    #endregion
}

/// <summary>
/// 眼睛状态
/// </summary>
[Description("眼睛状态")]
public enum EyeState
{
    /// <summary>
    /// 眼睛闭合
    /// </summary>
    [Description("闭眼")]
    Close,
    /// <summary>
    /// 眼睛张开
    /// </summary>
    [Description("睁眼")]
    Open,
    /// <summary>
    /// 不是眼睛区域
    /// </summary>
    [Description("不是眼睛区域")]
    Random,
    /// <summary>
    /// 状态无法判断
    /// </summary>
    [Description("无法判断")]
    Unknown
}
