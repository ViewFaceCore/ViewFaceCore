using ViewFaceCore.Native;

namespace ViewFaceCore.Configs
{
    /// <summary>
    /// 人脸跟踪器配置
    /// </summary>
    static internal class FaceInitConfig
    {
        private const string _modelPath = "./viewfacecore/models/";

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
                        ViewFaceNative.SetModelPath(_modelPath);
                        IsInitialized = true;
                    }
                }
            }
        }
    }
}
