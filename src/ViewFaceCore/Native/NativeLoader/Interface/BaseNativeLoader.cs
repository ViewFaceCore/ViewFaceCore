using System;
using System.Collections.Generic;
using System.Text;
using ViewFaceCore.Exceptions;

namespace ViewFaceCore.Native.NativeLoader.Interface
{
    public abstract class BaseNativeLoader : INativeLoader
    {
        /// <summary>
        /// ViewFaceBridge 的所有依赖库。(按照依赖顺序排列)
        /// </summary>
        protected HashSet<string> BaseLibraryNames = new HashSet<string>()
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

        public abstract string LibraryPath { get; set; }

        public abstract void Load();

        private bool TryCombinePath(out string path, params string[] paths)
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

        private string CombinePath(string basePath, string[] paths)
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

        /// <summary>
        /// 模型路径（避免重复去获取路径）
        /// </summary>
        private static string _modelsPath = null;

        private string GetModelsDefaultPath()
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


        /// <summary>
        /// 设置模型默认加载路径
        /// </summary>
        protected void SetModelsDefaultPath(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                path = GetModelsDefaultPath();
            }

        }

        public static void SetModelPath(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentNullException(nameof(path), "Model path can not null.");
            //to utf-8
            byte[] pathUtf8Bytes = Encoding.Convert(Encoding.Default, Encoding.UTF8, Encoding.Default.GetBytes(path));
            if (pathUtf8Bytes.Length > ViewFaceNative.MAX_PATH_LENGTH)
                throw new NotSupportedException($"The path is too long, not support path more than {ViewFaceNative.MAX_PATH_LENGTH} byte.");
            path = Encoding.UTF8.GetString(pathUtf8Bytes);

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                ViewFaceNative.SetModelPathWindows(path);
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                ViewFaceNative.SetModelPathLinux(Encoding.UTF8.GetBytes(path));
            else
                throw new PlatformNotSupportedException($"Unsupported system type: {RuntimeInformation.OSDescription}");

            if (!path.Equals(ViewFaceNative.GetModelPath()))
                throw new SeetaFaceModelException($"Set model path to '{path}' failed, failed to verify this path.");
        }
    }
}
