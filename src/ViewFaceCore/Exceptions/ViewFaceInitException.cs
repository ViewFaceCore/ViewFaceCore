using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ViewFaceCore.Exceptions
{
    /// <summary>
    /// 初始化异常
    /// </summary>
    public class ViewFaceInitException : Exception
    {
        public ViewFaceInitException(string message) : base(message)
        {

        }

        public ViewFaceInitException(string message, Exception innerException) : base(message, innerException)
        {

        }
    }
}
