using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ViewFaceCore.Demo.VideoForm.Attributes
{
    [AttributeUsage(AttributeTargets.Property)]
    public class IsReadonlyAttribute : Attribute
    {
        public IsReadonlyAttribute()
        {

        }

        public IsReadonlyAttribute(bool isReadonly)
        {
            IsReadonly = isReadonly;
        }

        public bool IsReadonly { get; set; } = false;
    }
}
