using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace ViewFaceCore.Model
{
    /// <summary>
    /// 活体检测状态
    /// </summary>
    public enum AntiSpoofingStatus
    {
        /// <summary>
        /// 错误或没有找到指定的人脸索引处的人脸
        /// </summary>
        [Description("错误或没有找到指定的人脸索引处的人脸")]
        Error = -1,
        /// <summary>
        /// 真实人脸
        /// </summary>
        [Description("真实人脸")]
        Real = 0,
        /// <summary>
        /// 攻击人脸（假人脸）
        /// </summary>
        [Description("攻击人脸（假人脸）")]
        Spoof = 1,
        /// <summary>
        /// 无法判断（人脸成像质量不好）
        /// </summary>
        [Description("无法判断（人脸成像质量不好）")]
        Fuzzy = 2,
        /// <summary>
        /// 正在检测
        /// </summary>
        [Description("正在检测")]
        Detecting = 3,
    };
}
