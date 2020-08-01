using System;
using System.Collections.Generic;
using System.Text;

namespace ViewFaceCore.Sharp.Exceptions
{
    /// <summary>
    /// 提取人脸特征值失败时引发的异常。
    /// </summary>
    public class ExtractException : Exception
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="message">错误信息</param>
        public ExtractException(string message) : base(message) { }
    }
}
