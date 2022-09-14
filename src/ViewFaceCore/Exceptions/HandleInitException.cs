using System;

namespace ViewFaceCore.Exceptions
{
    /// <summary>
    /// 句柄获取异常
    /// </summary>
    public class HandleInitException : Exception
    {
        /// <summary>
        /// 句柄获取异常
        /// </summary>
        /// <param name="message">异常信息</param>
        public HandleInitException(string message) : base(message)
        {

        }

        /// <summary>
        /// 句柄获取异常
        /// </summary>
        /// <param name="message">异常信息</param>
        /// <param name="innerException">引发的异常</param>
        public HandleInitException(string message, Exception innerException) : base(message, innerException)
        {

        }
    }
}
