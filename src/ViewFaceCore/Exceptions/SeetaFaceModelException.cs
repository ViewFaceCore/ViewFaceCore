namespace ViewFaceCore.Exceptions;

/// <summary>
/// 初始化模型异常
/// </summary>
public class SeetaFaceModelException : Exception
{
    /// <summary>
    /// 初始化模型异常
    /// </summary>
    /// <param name="message">异常信息</param>
    public SeetaFaceModelException(string message) : base(message)
    {

    }

    /// <summary>
    /// 初始化模型异常
    /// </summary>
    /// <param name="message">异常信息</param>
    /// <param name="innerException">引发的异常</param>
    public SeetaFaceModelException(string message, Exception innerException) : base(message, innerException)
    {

    }
}
