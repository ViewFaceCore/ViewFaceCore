using System;
using System.Collections.Generic;
using System.Text;

namespace ViewFaceCore.Native.NativeLoader.Interface
{
    public interface INativeLoader
    {
        string LibraryPath { get; }

        void Load();
    }
}
