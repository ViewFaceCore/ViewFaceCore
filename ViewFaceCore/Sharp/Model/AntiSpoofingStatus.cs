using System;
using System.Collections.Generic;
using System.Text;

namespace ViewFaceCore.Sharp.Model
{
    /// <summary>
    /// 活体检测状态
    /// </summary>
    public enum AntiSpoofingStatus
    {
        /// <summary>
        /// 真实人脸
        /// </summary>
        Real = 0,
        /// <summary>
        /// 攻击人脸（假人脸）
        /// </summary>
        Spoof = 1,
        /// <summary>
        /// 无法判断（人脸成像质量不好）
        /// </summary>
        Fuzzy = 2,
        /// <summary>
        /// 正在检测
        /// </summary>
        Detecting = 3,
    };
}
