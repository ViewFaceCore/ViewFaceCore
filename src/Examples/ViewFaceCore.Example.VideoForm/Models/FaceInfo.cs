using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ViewFaceCore.Model;

namespace ViewFaceCore.Demo.VideoForm.Models
{
    internal class FaceInfo
    {
        public int Pid { get; set; }

        public bool HasMask { get; set; }

        public FaceRect Location { get; set; }

        public RectangleF Rectangle
        {
            get
            {
                return new RectangleF(this.Location.X, this.Location.Y, this.Location.Width, this.Location.Height);
            }
        }

        public int Age { get; set; }

        public Gender Gender { get; set; }

        public string GenderDescribe
        {
            get
            {
                switch (this.Gender)
                {
                    case Gender.Male:
                        return "男";
                    case Gender.Female:
                        return "女";
                    case Gender.Unknown:
                        return "未知";
                }
                return "未知";
            }
        }

        public string Name { get; set; }
    }
}
