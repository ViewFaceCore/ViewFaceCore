using System;
using System.Collections.Generic;
using System.Text;

namespace ViewFaceCore.Sharp.Model
{
    /// <summary>
    /// 人脸关键点类型
    /// </summary>
    public enum MarkType : int
    {
        /// <summary>
        /// 68个关键点。
        /// </summary>
        Normal = 0,
        /// <summary>
        /// 戴口罩的关键点。
        /// </summary>
        Mask,
        /// <summary>
        /// 5个关键点。
        /// </summary>
        Light,
    }
}
