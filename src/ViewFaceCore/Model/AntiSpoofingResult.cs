using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ViewFaceCore.Model
{
    public class AntiSpoofingResult
    {
        public AntiSpoofingResult(AntiSpoofingStatus status, float clarity, float reality)
        {
            this.Status = status;
            this.Clarity = clarity;
            this.Reality = reality;
        }

        /// <summary>
        /// 活体检测状态
        /// </summary>
        public AntiSpoofingStatus Status { get; set; }

        /// <summary>
        /// 清晰度
        /// </summary>
        public float Clarity { get; set; }

        /// <summary>
        /// 真实度
        /// </summary>
        public float Reality { get; set; }
    }
}
