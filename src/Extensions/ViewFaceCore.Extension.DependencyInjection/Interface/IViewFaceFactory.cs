using System;
using System.Collections.Generic;
using System.Text;
using ViewFaceCore.Core;

namespace ViewFaceCore.Extension.DependencyInjection
{
    public interface IViewFaceFactory
    {
        T Get<T>() where T : IViewFace;
    }
}
