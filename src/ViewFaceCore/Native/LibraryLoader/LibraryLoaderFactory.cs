using ViewFaceCore.Native.LibraryLoader.Interface;
using ViewFaceCore.Native.LibraryLoader.LibraryLoaders;

namespace ViewFaceCore.Native.LibraryLoader
{
    internal class LibraryLoaderFactory
    {
        public static ILibraryLoader Create()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return new WinLibraryLoader();
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) || RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                return new UnixLibraryLoader();
            }
            else
            {
                throw new PlatformNotSupportedException($"Unsupported operating system platform: {RuntimeInformation.OSDescription}");
            }
        }
    }
}
