#if !NET45_OR_GREATER // net40 »ò¸üµÍ°æ±¾

using System.Runtime.InteropServices;

namespace System.Runtime.CompilerServices
{
    /// <summary>
    /// Used with Task(of void)
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Size = 1)]
    internal struct VoidTaskResult
    {
    }
}

#endif