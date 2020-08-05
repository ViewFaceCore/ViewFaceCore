
using System;
using System.Collections.Generic;
using System.Text;

namespace ViewFaceCore.Sharp.Model
{
    /// <summary>
    /// 质量评估结果
    /// </summary>
    public class QualityResult
    {
        /// <summary>
        /// 质量评估等级
        /// </summary>
        public QualityLevel Level { get; set; } = QualityLevel.Low;
        /// <summary>
        /// 质量评估分数
        /// <para>越大越好，没有范围限制</para>
        /// </summary>
        public float Score { get; set; } = 0;
    }
}
