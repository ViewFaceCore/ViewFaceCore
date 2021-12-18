using System;
using ViewFaceCore.Sharp.Model;
using System.Runtime.InteropServices;
using System.IO;
using System.Reflection;

namespace ViewFaceCore.Plus
{
    /// <summary>
    /// 适用于 Any CPU 的 ViewFacePlus
    /// </summary>
    static partial class ViewFaceBridge
    {
        [DllImport("kernel32", CharSet = CharSet.Auto, SetLastError = true)]
        public static extern bool SetDllDirectory(string path);

        static ViewFaceBridge()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                //TODO: 在 Linux 下搜索动态库方案
                if (!Environment.Is64BitOperatingSystem || !Environment.Is64BitProcess)
                {
                    throw new Exception("Not support 32 bit linux system!");
                }

                string architecture = "x64";
                switch (RuntimeInformation.ProcessArchitecture)
                {
                    case Architecture.X86:
                        throw new Exception("Not support 32 bit linux system!");
                        architecture = "x86"; break;
                    case Architecture.Arm:
                        throw new Exception("Not support 32 bit linux system!");
                        architecture = "arm"; break;
                    case Architecture.Arm64:
                        throw new Exception("Not support ARM64 linux system!");
                        architecture = "arm64"; break;
                    case Architecture.X64: architecture = "x64"; break;
                }
                string val = Environment.GetEnvironmentVariable("LD_LIBRARY_PATH");
                if (!string.IsNullOrEmpty(val)) { val += ","; }
                Environment.SetEnvironmentVariable("LD_LIBRARY_PATH", $"{val}{GetFullPath($"viewfacecore/linux/{architecture}/libViewFaceBridge.so")}/:$LD_LIBRARY_PATH");
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                string architecture = Environment.Is64BitProcess ? "x64" : "x86";
                SetDllDirectory(GetFullPath($"viewfacecore/win/{architecture}/ViewFaceBridge.dll"));
            }
        }

        private static string GetFullPath(string path)
        {
            if (!File.Exists(path))
            {
                throw new Exception("Lib not exist!");
            }
            FileInfo fileInfo = new FileInfo(path);
            return fileInfo.DirectoryName;
        }

        /// <summary>
        /// 获取一个值，指示当前运行的进程是否是 64位
        /// </summary>
        public static bool Is64BitProcess { get; } = Environment.Is64BitProcess;

        /// <summary>
        /// 获取或设置人脸模型目录
        /// </summary>
        public static string ModelPath { get => GetModelPath(); set => SetModelPath(value); }

    }
}
