using System;
using System.Runtime.InteropServices;
using System.IO;
using System.Reflection;
using System.Collections.Generic;

namespace ViewFaceCore.Native
{
    /// <summary>
    /// 适用于 Any CPU 的 ViewFacePlus
    /// </summary>
    internal static partial class ViewFaceNative
    {
        [DllImport("kernel32", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern bool SetDllDirectory(string path);

        /// <summary>
        /// 获取本机库目录
        /// </summary>
        private static string LibraryPath
        {
            get
            {
                string architecture, platform;
                switch (RuntimeInformation.ProcessArchitecture)
                {
                    case Architecture.X86: architecture = "x86"; break;
                    case Architecture.X64: architecture = "x64"; break;
                    case Architecture.Arm: architecture = "arm"; break;
                    case Architecture.Arm64: architecture = "arm64"; break;
                    default: throw new PlatformNotSupportedException($"不支持的处理器体系结构: {RuntimeInformation.ProcessArchitecture}");
                }
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                { platform = "win"; }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                { platform = "linux"; }
                else
                { throw new PlatformNotSupportedException($"不支持的操作系统: {RuntimeInformation.OSDescription}"); }

                var libraryPath = Path.Combine(Path.GetDirectoryName(Assembly.GetAssembly(typeof(ViewFace)).Location), "viewfacecore", platform, architecture);
                if (Directory.Exists(libraryPath))
                { return libraryPath; }
                else { throw new DirectoryNotFoundException($"找不到本机库目录: {libraryPath}"); }
            }
        }

        /// <summary>
        /// ViewFaceBridge 的所有依赖库。(按照依赖顺序排列)
        /// </summary>
        private static readonly List<string> Libraries = new List<string>()
        {
            "SeetaAuthorize",
            "tennis",
            "tennis_haswell",
            "tennis_pentium",
            "tennis_sandy_bridge",
            "SeetaMaskDetector200",
            "SeetaAgePredictor600",
            "SeetaEyeStateDetector200",
            "SeetaFaceAntiSpoofingX600",
            "SeetaFaceDetector600",
            "SeetaFaceLandmarker600",
            "SeetaFaceRecognizer610",
            "SeetaFaceTracking600",
            "SeetaGenderPredictor600",
            "SeetaPoseEstimation600",
            "SeetaQualityAssessor300",
        };

        /// <summary>
        /// 在首次使用时初始化本机库目录。
        /// <para>贡献: <a href="https://github.com/withsalt">withsalt</a></para>
        /// <para>参考: <a href="https://docs.microsoft.com/en-us/dotnet/standard/native-interop/cross-platform">Cross Platform P/Invoke</a></para>
        /// <para></para>
        /// </summary>
        /// <exception cref="BadImageFormatException"></exception>
        /// <exception cref="FileNotFoundException"></exception>
        /// <exception cref="PlatformNotSupportedException"></exception>
        static ViewFaceNative()
        {
#if NETFRAMEWORK || NETSTANDARD
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            { SetDllDirectory(LibraryPath); }
            else
            { throw new PlatformNotSupportedException($"不支持的操作系统: {RuntimeInformation.OSDescription}"); }
#elif NETCOREAPP3_1_OR_GREATER
            #region Resolver Libraries on Linux
            // Author: <a href="https://github.com/withsalt">withsalt</a>
            // 预加载 ViewFaceBridge 的所有依赖库

            string format;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            { format = "{0}.dll"; }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            { format = "lib{0}.so"; }
            else
            { throw new PlatformNotSupportedException($"不支持的操作系统: {RuntimeInformation.OSDescription}"); }

            foreach (var library in Libraries)
            {
                string libraryPath = Path.Combine(LibraryPath, string.Format(format, library));
                if (File.Exists(libraryPath))
                {
                    if (NativeLibrary.Load(libraryPath) == IntPtr.Zero)
                    { throw new BadImageFormatException($"加载本机库失败: {library}"); }
                }
                else if(!libraryPath.Contains("tennis_"))
                { throw new FileNotFoundException($"找不到本机库：{libraryPath}"); }
            }

            NativeLibrary.SetDllImportResolver(Assembly.GetExecutingAssembly(), (libraryName, assembly, searchPath) =>
            {
                var library = "ViewFaceBridge";
                if (libraryName.Equals(library, StringComparison.OrdinalIgnoreCase))
                {
                    string libraryPath = Path.Combine(LibraryPath, string.Format(format, library));
                    return NativeLibrary.Load(libraryPath, assembly, searchPath ?? DllImportSearchPath.AssemblyDirectory);
                }
                return IntPtr.Zero;
            });
            #endregion
#else
            throw new PlatformNotSupportedException($"不支持的 .NET 平台: {RuntimeInformation.FrameworkDescription}");
#endif
        }

    }
}
