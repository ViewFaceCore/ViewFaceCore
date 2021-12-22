using System;
using ViewFaceCore.Sharp.Model;
using System.Runtime.InteropServices;
using System.IO;
using System.Reflection;
using System.Collections.Generic;
using System.Linq;

namespace ViewFaceCore.Plus
{
    /// <summary>
    /// 适用于 Any CPU 的 ViewFacePlus
    /// </summary>
    static partial class ViewFaceBridge
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

                var libraryPath = Path.Combine(Environment.CurrentDirectory, "viewfacecore", platform, architecture);
                if (Directory.Exists(libraryPath))
                { return libraryPath; }
                else { throw new DirectoryNotFoundException($"找不到本机库目录: {libraryPath}"); }
            }
        }

        /// <summary>
        /// Linux 下 libViewFaceBridge.so 的所有依赖库。(按照依赖顺序排列)
        /// </summary>
        private static readonly List<string> Libraries = new List<string>()
        {
            "libSeetaAuthorize.so",
            "libtennis.so",
            "libtennis_haswell.so",
            "libtennis_pentium.so",
            "libtennis_sandy_bridge.so",
            "libSeetaMaskDetector200.so",
            "libSeetaAgePredictor600.so",
            "libSeetaEyeStateDetector200.so",
            "libSeetaFaceAntiSpoofingX600.so",
            "libSeetaFaceDetector600.so",
            "libSeetaFaceLandmarker600.so",
            "libSeetaFaceRecognizer610.so",
            "libSeetaFaceTracking600.so",
            "libSeetaGenderPredictor600.so",
            "libSeetaPoseEstimation600.so",
            "libSeetaQualityAssessor300.so",
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
        static ViewFaceBridge()
        {
#if NETFRAMEWORK || NETSTANDARD
            SetDllDirectory(LibraryPath);
#elif NETCOREAPP3_1_OR_GREATER
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                SetDllDirectory(LibraryPath);
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                #region Resolver Libraries on Linux
                // Author: <a href="https://github.com/withsalt">withsalt</a>
                // 预加载 libViewFaceBridge.so 的所有依赖库
                foreach (var library in Libraries)
                {
                    string libraryPath = Path.Combine(LibraryPath, library);
                    if (File.Exists(libraryPath))
                    {
                        if (NativeLibrary.Load(library) == IntPtr.Zero)
                        { throw new BadImageFormatException($"加载本机库失败: {library}"); }
                    }
                    else
                    { throw new FileNotFoundException($"找不到本机库：{libraryPath}"); }
                }

                NativeLibrary.SetDllImportResolver(Assembly.GetExecutingAssembly(), (libraryName, assembly, searchPath) =>
                {
                    if (libraryName.Equals("ViewFaceBridge", StringComparison.OrdinalIgnoreCase))
                    {
                        return NativeLibrary.Load(Path.Combine(LibraryPath, "libViewFaceBridge.so"), assembly, searchPath);
                    }
                    return IntPtr.Zero;
                });
                #endregion
            }
            else
            { throw new PlatformNotSupportedException($"不支持的操作系统: {RuntimeInformation.OSDescription}"); }
#else
            throw new PlatformNotSupportedException($"不支持的 .NET 平台: {RuntimeInformation.FrameworkDescription}");
#endif
        }

    }
}
