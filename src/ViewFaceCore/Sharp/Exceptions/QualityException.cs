using System;
using ViewFaceCore.Sharp.Model;

namespace ViewFaceCore.Sharp.Exceptions
{
    /// <summary>
    /// 质量评估失败时引发的异常
    /// </summary>
    public class QualityException : Exception
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="message"></param>
        /// <param name="type"></param>
        public QualityException(string message, QualityType type) : base(message) { Type = type; }

        /// <summary>
        /// 产生错误的质量评估模块类型
        /// </summary>
        public QualityType Type { get; }
    }
}
