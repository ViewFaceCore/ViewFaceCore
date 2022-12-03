using System;
using System.Runtime.InteropServices;

namespace ViewFaceCore.Native.LibraryLoader.LibraryLoaders.Platforms
{
    /// <summary>
    /// 
    /// sudo apt install -y --allow-unauthenticated libc6-dev libx11-dev
    /// </summary>
    internal static class Libdl
    {
        private const string LibName = "libdl";

        public const int RTLD_NOW = 0x002;

        [DllImport(LibName)]
        public static extern IntPtr dlopen(string fileName, int flags);

        [DllImport(LibName)]
        public static extern IntPtr dlsym(IntPtr handle, string name);

        [DllImport(LibName)]
        public static extern int dlclose(IntPtr handle);

        [DllImport(LibName)]
        public static extern string dlerror();
    }
}
