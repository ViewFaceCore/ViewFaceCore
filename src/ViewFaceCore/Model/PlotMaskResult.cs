using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ViewFaceCore.Model
{
    /// <summary>
    /// 戴口罩识别返回模型
    /// </summary>
    public class PlotMaskResult
    {
        public PlotMaskResult(float score, bool status, bool masked)
        {
            this.Score = score;
            this.Status = status;
            this.Masked = masked;
        }

        /// <summary>
        /// 评估分数
        /// </summary>
        public float Score { get; set; }

        /// <summary>
        /// 是否检测成功
        /// </summary>
        public bool Status { get; set; }

        /// <summary>
        /// 是否戴口罩
        /// </summary>
        public bool Masked { get; set; }
    }
}
