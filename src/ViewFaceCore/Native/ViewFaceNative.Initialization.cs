using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using ViewFaceCore.Configs;

#if NETCOREAPP3_1_OR_GREATER
using System.Runtime.Intrinsics.X86;
#endif

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
                    default: throw new PlatformNotSupportedException($"Unsupported processor architecture: {RuntimeInformation.ProcessArchitecture}");
                }
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                    platform = "win";
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                    platform = "linux";
                else
                    throw new PlatformNotSupportedException($"Unsupported system type: {RuntimeInformation.OSDescription}");
                if (!TryCombinePath(out string libraryPath, "viewfacecore", platform, architecture))
                    throw new DirectoryNotFoundException("Can not found library path.");
                _libraryPath = libraryPath;
                return _libraryPath;
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
                if (TryCombinePath(out string modelsPath, "viewfacecore", "models"))
                {
                    _modelsPath = modelsPath;
                    return modelsPath;
                }
                throw new DirectoryNotFoundException("Can not found models path.");
            }
        }

        /// <summary>
        /// ViewFaceBridge 的所有依赖库。(按照依赖顺序排列)
        /// </summary>
        private static List<string> Libraries = new List<string>()
        {
            "tennis",
            "tennis_haswell",
            "tennis_pentium",
            "tennis_sandy_bridge",
            "SeetaAuthorize",
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
            //根据指令集，设置Tennis依赖库
            ResetTennisDependency();
#if NETFRAMEWORK || NETSTANDARD
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            { SetDllDirectory(LibraryPath); }
            else
            { throw new PlatformNotSupportedException($"Unsupported system type: {RuntimeInformation.OSDescription}"); }
#elif NETCOREAPP3_1_OR_GREATER
            foreach (var library in Libraries)
            {
                //不支持Avx2
                if (!Avx2.IsSupported && (library.Contains("tennis_haswell") || library.Contains("tennis_sandy_bridge"))) continue;
                //不支持Fma
                if (!Fma.IsSupported && library.Contains("tennis_sandy_bridge")) continue;
                //Combine Library Path
                string libraryPath = GetLibraryFullName(library);
                if (!File.Exists(libraryPath))
                {
                    if (library.Contains("tennis_", StringComparison.OrdinalIgnoreCase))
                        continue;
                    throw new FileNotFoundException($"Can not found library {libraryPath}.");
                }
                if (NativeLibrary.Load(libraryPath) == IntPtr.Zero)
                {
                    throw new BadImageFormatException($"Can not load native library {libraryPath}.");
                }
            }

            NativeLibrary.SetDllImportResolver(Assembly.GetAssembly(typeof(ViewFaceNative)), (libraryName, assembly, searchPath) =>
            {
                if (!libraryName.Equals(BRIDGE_LIBRARY_NAME, StringComparison.OrdinalIgnoreCase))
                    return IntPtr.Zero;
                string libraryPath = GetLibraryFullName(BRIDGE_LIBRARY_NAME);
                return NativeLibrary.Load(libraryPath, assembly, searchPath ?? DllImportSearchPath.AssemblyDirectory);
            });
#else
            throw new PlatformNotSupportedException($"Unsupported .net runtime {RuntimeInformation.FrameworkDescription}");
#endif
            //设置模型位置
            SetModelPath(ModelsPath);
        }

        public static string GetLibraryPath() => LibraryPath;

        private static bool TryCombinePath(out string path, params string[] paths)
        {
            string[] prepareCombinePaths = new string[paths.Length + 1];
            for (int i = 0; i < paths.Length; i++)
            {
                prepareCombinePaths[i + 1] = paths[i];
            }
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                //windiws环境下，如果当前进程名称包含iis或者w3wp，优先返回
                string processName = Process.GetCurrentProcess().ProcessName;
                if (!string.IsNullOrEmpty(processName)
                    && (processName.IndexOf("iis", StringComparison.OrdinalIgnoreCase) >= 0
                    || processName.IndexOf("w3wp", StringComparison.OrdinalIgnoreCase) >= 0))
                {
                    path = CombinePath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "bin"), prepareCombinePaths);
                    if (!string.IsNullOrWhiteSpace(path))
                    {
                        return true;
                    }
                }
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
            path = CombinePath(Path.GetDirectoryName(Assembly.GetAssembly(typeof(ViewFaceNative)).Location), prepareCombinePaths);
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

        private static void ResetTennisDependency()
        {
            //Arm不需要处理
            if (RuntimeInformation.ProcessArchitecture != Architecture.X86
                && RuntimeInformation.ProcessArchitecture != Architecture.X64)
            {
                return;
            }
            switch (GlobalConfig.X86Instruction)
            {
                case X86Instruction.AVX2:
                    {
                        //只支持AVX2
                        GlobalConfig.WriteLog("CPU only support AVX2 instruction, will use tennis_sandy_bridge.");

                        List<string> removeLibs = new List<string>() { "tennis_haswell", "tennis_pentium" };
                        removeLibs.ForEach(p =>
                        {
                            if (Libraries.Contains(p))
                            {
                                Libraries.Remove(p);
                            }
                        });
                        string supportTennisLibPath = GetLibraryFullName("tennis_sandy_bridge");
                        if (!File.Exists(supportTennisLibPath))
                        {
                            return;
                        }
                        string baseTennisLibPath = GetLibraryFullName("tennis");
                        if (File.Exists(supportTennisLibPath))
                        {
                            File.Delete(baseTennisLibPath);
                        }
                        File.Copy(supportTennisLibPath, baseTennisLibPath, true);
                    }
                    break;
                case X86Instruction.SSE2:
                    {
                        //只支持SSE2
                        GlobalConfig.WriteLog("CPU only support SSE2 instruction, will use tennis_pentium.");

                        List<string> removeLibs = new List<string>() { "tennis_haswell", "tennis_sandy_bridge" };
                        removeLibs.ForEach(p =>
                        {
                            if (Libraries.Contains(p))
                            {
                                Libraries.Remove(p);
                            }
                        });
                        string supportTennisLibPath = GetLibraryFullName("tennis_pentium");
                        if (!File.Exists(supportTennisLibPath))
                        {
                            return;
                        }
                        string baseTennisLibPath = GetLibraryFullName("tennis");
                        if (File.Exists(supportTennisLibPath))
                        {
                            File.Delete(baseTennisLibPath);
                        }
                        File.Copy(supportTennisLibPath, baseTennisLibPath, true);
                    }
                    break;
            }
        }

        private static string GetLibraryFullName(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentNullException("name can not null", nameof(name));
            }
            string format;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                format = "{0}.dll";
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                format = "lib{0}.so";
            else
                throw new PlatformNotSupportedException($"Unsupported system type: {RuntimeInformation.OSDescription}");
            return Path.Combine(LibraryPath, string.Format(format, name));
        }
    }
}
