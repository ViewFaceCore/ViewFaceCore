using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ViewFaceCore.Demo.VideoForm.Attributes
{
    [AttributeUsage(AttributeTargets.Property)]
    public class IsIgnoreAttribute : Attribute
    {
        public IsIgnoreAttribute()
        {

        }

        public IsIgnoreAttribute(bool isIgnore)
        {
            IsIgnore = isIgnore;
        }

        public bool IsIgnore { get; set; } = true;
    }
}
