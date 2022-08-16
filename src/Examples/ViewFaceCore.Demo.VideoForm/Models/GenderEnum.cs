using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ViewFaceCore.Demo.VideoForm.Models
{
    public enum GenderEnum
    {
        [Description("男")]
        Male = 1,

        [Description("女")]
        Female = 2,

        [Description("未知")]
        Unknown = -1
    }
}
