using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using ViewFaceCore.Configs.Enums;
using ViewFaceCore.Native.LibraryLoader.Interface;

namespace ViewFaceCore.Native.LibraryLoader.PathResolvers
{
    internal class DefaultPathResolver : BasePathResolver
    {
        private string _modelsPath = null;
        private string _libraryPath = null;

        private const string DEFAULT_LIBRARY_PATH = "viewfacecore";
        private const string DEFAULT_MODELS_PATH = "models";

        public DefaultPathResolver(DeviceType deviceType) : base(deviceType)
        {

        }

        public override string GetModelsPath()
        {
            if (!string.IsNullOrEmpty(_modelsPath))
            {
                return _modelsPath;
            }
            if (TryCombine(out string modelsPath, DEFAULT_LIBRARY_PATH, DEFAULT_MODELS_PATH))
            {
                _modelsPath = modelsPath;
                return modelsPath;
            }
            throw new DirectoryNotFoundException("Can not found default models path.");
        }

        public override string GetLibraryPath()
        {
            if (!string.IsNullOrEmpty(_libraryPath))
                return _libraryPath;

            string architecture = RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.X86 => "x86",
                Architecture.X64 => "x64",
                Architecture.Arm => "arm",
                Architecture.Arm64 => "arm64",
                _ => throw new PlatformNotSupportedException($"Unsupported processor architecture: {RuntimeInformation.ProcessArchitecture}"),
            };

            string platform;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                platform = "win";
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                platform = "linux";
            else
                throw new PlatformNotSupportedException($"Unsupported system type: {RuntimeInformation.OSDescription}");

            if (!TryCombine(out string libraryPath, "viewfacecore", platform, architecture))
                throw new DirectoryNotFoundException("Can not found library path.");

            _libraryPath = libraryPath;
            return _libraryPath;
        }

        #region private

        private bool TryCombine(out string path, params string[] paths)
        {
            string[] prepareCombinePaths = new string[paths.Length + 1];
            for (int i = 0; i < paths.Length; i++)
            {
                prepareCombinePaths[i + 1] = paths[i];
            }
            if (IsDeployByIIS())
            {
                path = PathCombine(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "bin"), prepareCombinePaths);
                if (!string.IsNullOrWhiteSpace(path))
                {
                    return true;
                }
            }
            path = PathCombine(AppDomain.CurrentDomain.BaseDirectory, prepareCombinePaths);
            if (!string.IsNullOrWhiteSpace(path))
            {
                return true;
            }
            path = PathCombine(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "bin"), prepareCombinePaths);
            if (!string.IsNullOrWhiteSpace(path))
            {
                return true;
            }
            path = PathCombine(Path.GetDirectoryName(Assembly.GetAssembly(typeof(ViewFaceNative)).Location), prepareCombinePaths);
            if (!string.IsNullOrWhiteSpace(path))
            {
                return true;
            }
            return false;
        }

        private string PathCombine(string basePath, string[] paths)
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

        private bool IsDeployByIIS()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                //windiws环境下，如果当前进程名称包含iis或者w3wp，优先返回
                string processName = Process.GetCurrentProcess().ProcessName;
                if (!string.IsNullOrEmpty(processName)
                    && (processName.IndexOf("iis", StringComparison.OrdinalIgnoreCase) >= 0
                    || processName.IndexOf("w3wp", StringComparison.OrdinalIgnoreCase) >= 0))
                {
                    return true;
                }
            }
            return false;
        }


        #endregion
    }
}
