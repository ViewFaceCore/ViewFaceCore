using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using ViewFaceCore.Configs;
using ViewFaceCore.Core;

namespace ViewFaceCore.Native
{
    /// <summary>
    /// 适用于 Any CPU 的 ViewFacePlus
    /// </summary>
    internal static partial class ViewFaceNative
    {
        [DllImport("kernel32", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern bool SetDllDirectory(string path);

        private static string _libraryPath = null;

        /// <summary>
        /// 获取本机库目录
        /// </summary>
        private static string LibraryPath
        {
            get
            {
                if (!string.IsNullOrEmpty(_libraryPath))
                    return _libraryPath;
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
                    platform = "win";
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                    platform = "linux";
                else
                    throw new PlatformNotSupportedException($"不支持的操作系统: {RuntimeInformation.OSDescription}");
                string libraryPath;
                if (!TryCombinePath(out libraryPath, "viewfacecore", platform, architecture))
                    throw new DirectoryNotFoundException("Not found library path.");
                if (Directory.Exists(libraryPath))
                {
                    _libraryPath = libraryPath;
                    return _libraryPath;
                }
                else
                    throw new DirectoryNotFoundException($"找不到本机库目录: {libraryPath}");
            }
        }

        /// <summary>
        /// 模型路径（避免重复去获取路径）
        /// </summary>
        private static string _modelsPath = null;

        private static string ModelsPath
        {
            get
            {
                if (!string.IsNullOrEmpty(_modelsPath))
                    return _modelsPath;
                string modelsPath;
                if (TryCombinePath(out modelsPath, "viewfacecore", "models"))
                {
                    _modelsPath = modelsPath;
                    return modelsPath;
                }
                throw new DirectoryNotFoundException("Not found models path.");
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
            //设置模型位置
            ViewFaceNative.SetModelPath(ModelsPath);
        }

        private static bool TryCombinePath(out string path, params string[] paths)
        {
            string[] prepareCombinePaths = new string[paths.Length + 1];
            for (int i = 0; i < paths.Length; i++)
            {
                prepareCombinePaths[i + 1] = paths[i];
            }
            path = CombinePath(Path.GetDirectoryName(Assembly.GetAssembly(typeof(ViewFaceNative)).Location), prepareCombinePaths);
            if (!string.IsNullOrWhiteSpace(path))
            {
                return true;
            }
            path = CombinePath(AppDomain.CurrentDomain.BaseDirectory, prepareCombinePaths);
            if (!string.IsNullOrWhiteSpace(path))
            {
                return true;
            }
            path = CombinePath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "bin"), prepareCombinePaths);
            if (!string.IsNullOrWhiteSpace(path))
            {
                return true;
            }
            return false;
        }

        private static string CombinePath(string basePath, string[] paths)
        {
            if (paths == null || paths.Length < 1)
            {
                return null;
            }
            paths[0] = basePath;
            string outPath = Path.Combine(paths) + Path.DirectorySeparatorChar;
            if (Directory.Exists(outPath))
            {
                return outPath;
            }
            return null;
        }

    }
}
