using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ViewFaceCore.Configs
{
    public class MaskDetectConfig : BaseConfig
    {
        /// <summary>
        /// 阈值，默认0.5
        /// </summary>
        /// <remarks>
        /// 一般性的，score超过0.5，则认为是检测带上了口罩
        /// </remarks>
        public float Threshold { get; set; } = 0.5f;

        /// <summary>
        /// 设置阈值
        /// </summary>
        /// <param name="threshold"></param>
        /// <returns></returns>
        public MaskDetectConfig SetThreshold(float threshold)
        {
            this.Threshold = threshold;
            return this;
        }
    }
}
