namespace ViewFaceCore.Exceptions;

/// <summary>
/// 初始化模型异常
/// </summary>
public class LoadModelException : Exception
{
    /// <summary>
    /// 初始化模型异常
    /// </summary>
    /// <param name="message">异常信息</param>
    public LoadModelException(string message) : base(message)
    {

    }

    /// <summary>
    /// 初始化模型异常
    /// </summary>
    /// <param name="message">异常信息</param>
    /// <param name="innerException">引发的异常</param>
    public LoadModelException(string message, Exception innerException) : base(message, innerException)
    {

    }
}
