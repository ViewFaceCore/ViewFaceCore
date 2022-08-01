using System.IO;
using System.Reflection;
using ViewFaceCore.Native;

namespace ViewFaceCore.Configs
{
    /// <summary>
    /// 人脸跟踪器配置
    /// </summary>
    static internal class FaceGlobalConfig
    {
        private static string ModelPath
        {
            get
            {
                return Path.Combine(BasePath, "viewfacecore", "models") + Path.DirectorySeparatorChar;
            }
        }

        public static string BasePath
        {
            get
            {
                return Path.GetDirectoryName(Assembly.GetAssembly(typeof(FaceGlobalConfig)).Location);
            }
        }

        public static bool IsInitialized { get; private set; } = false;

        private readonly static object _initLocker = new object();

        public static void Init()
        {
            if (!IsInitialized)
            {
                lock (_initLocker)
                {
                    if (!IsInitialized)
                    {
                        ViewFaceNative.SetModelPath(ModelPath);
                        IsInitialized = true;
                    }
                }
            }
        }
    }
}
