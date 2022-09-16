namespace ViewFaceCore.Models;

/// <summary>
/// 人脸跟踪信息
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct FaceTrackInfo : IFormattable
{
    private readonly FaceRect pos;
    private readonly float score;

    private readonly int frame_no;
    private readonly int PID;
    private readonly int step;

    /// <summary>
    /// 人脸位置
    /// </summary>
    public FaceRect Location => pos;
    /// <summary>
    /// 人脸置信度
    /// </summary>
    public float Score => score;
    /// <summary>
    /// 人脸标识Id
    /// </summary>
    public int Pid => PID;

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
        string stips = nameof(Score), ltips = nameof(Location), ptips = nameof(Pid);

        if ((formatProvider ?? Thread.CurrentThread.CurrentCulture) is CultureInfo cultureInfo && cultureInfo.Name.StartsWith("zh"))
        { stips = "置信度"; ltips = "位置"; ptips = "标识Id"; }

        return $"{{{ptips}:{Pid}, {stips}:{Score}, {ltips}:{Location.ToString(format, formatProvider)}}}";
    }
    #endregion

    /// <summary>
    /// 隐式转换
    /// </summary>
    /// <param name="value"></param>
    public static implicit operator FaceInfo(FaceTrackInfo value) => new(value.Location, value.Score);
}
