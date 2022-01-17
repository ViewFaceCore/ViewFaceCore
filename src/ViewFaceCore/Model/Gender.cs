using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace ViewFaceCore.Model
{
    /// <summary>
    /// 人脸性别
    /// </summary>
    public enum Gender
    {
        /// <summary>
        /// 未知 或 识别失败。
        /// </summary>
        [Description("未知")]
        Unknown = -1,
        /// <summary>
        /// 男性
        /// </summary>
        [Description("男")] 
        Male = 0,
        /// <summary>
        /// 女性
        /// </summary>
        [Description("女")]
        Female = 1
    }
}
