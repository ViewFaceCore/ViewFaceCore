using System;

namespace ViewFaceCore.Exceptions
{
    /// <summary>
    /// 获取人脸关键点失败时引发的异常
    /// </summary>
    public class MarkException : Exception
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="message">错误信息</param>
        public MarkException(string message) : base(message) { }
    }
}
