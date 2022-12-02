namespace ViewFaceCore.Exceptions;

/// <summary>
/// 句柄获取异常
/// </summary>
public class ModuleInitializeException : Exception
{
    /// <summary>
    /// 句柄获取异常
    /// </summary>
    /// <param name="message">异常信息</param>
    public ModuleInitializeException(string message) : base(message)
    {

    }

    /// <summary>
    /// 句柄获取异常
    /// </summary>
    /// <param name="message">异常信息</param>
    /// <param name="innerException">引发的异常</param>
    public ModuleInitializeException(string message, Exception innerException) : base(message, innerException)
    {

    }
}
