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
        public PlotMaskResult(float score, bool status)
        {
            this.Score = score;
            this.Status = status;
        }

        /// <summary>
        /// 评估分数
        /// </summary>
        public float Score { get; set; }

        /// <summary>
        /// 是否戴口罩
        /// </summary>
        public bool Status { get; set; }
    }
}
