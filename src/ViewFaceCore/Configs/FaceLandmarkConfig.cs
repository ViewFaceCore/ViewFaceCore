using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ViewFaceCore.Model;

namespace ViewFaceCore.Configs
{
    public class FaceLandmarkConfig : BaseConfig
    {
        /// <summary>
        /// 关键点类型
        /// </summary>
        public MarkType MarkType { get; set; } = MarkType.Light;

        public FaceLandmarkConfig()
        {

        }

        public FaceLandmarkConfig(MarkType markType)
        {
            MarkType = markType;
        }
    }
}
