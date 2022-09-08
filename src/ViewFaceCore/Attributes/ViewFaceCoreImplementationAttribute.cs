using System;
using System.Collections.Generic;
using System.Text;

namespace ViewFaceCore.Attributes
{
    [AttributeUsage(AttributeTargets.Assembly | AttributeTargets.Class, Inherited = false, AllowMultiple = true)]
    internal sealed class ViewFaceCoreImplementationAttribute : Attribute
    {
        readonly Type imageType;

        public Type ImageType { get => imageType; }

        public ViewFaceCoreImplementationAttribute(Type imageType)
        {
            this.imageType = imageType;
        }
    }
}
