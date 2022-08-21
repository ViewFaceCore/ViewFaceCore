using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ViewFaceCore.Model;

namespace ViewFaceCore.Configs
{
    public class FaceLandmarkConfig : BaseConfig
    {
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
