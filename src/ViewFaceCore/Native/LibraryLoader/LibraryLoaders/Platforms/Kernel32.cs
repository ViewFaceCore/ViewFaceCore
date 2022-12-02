using System;
using System.Collections.Generic;
using System.Text;

namespace ViewFaceCore.Native.LibraryLoader.LibraryLoaders.Platforms
{
    internal static class Kernel32
    {
        [DllImport("kernel32", CharSet = CharSet.Auto, SetLastError = true)]
        public static extern IntPtr LoadLibrary(string fileName);

        [DllImport("kernel32", CharSet = CharSet.Auto, SetLastError = true)]
        public static extern IntPtr GetProcAddress(IntPtr module, string procName);

        [DllImport("kernel32", CharSet = CharSet.Auto, SetLastError = true)]
        public static extern int FreeLibrary(IntPtr module);

        [DllImport("kernel32", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern bool SetDllDirectory(string path);
    }
}
