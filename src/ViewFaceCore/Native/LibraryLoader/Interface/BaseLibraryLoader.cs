using ViewFaceCore.Configs;
using ViewFaceCore.Configs.Enums;
using ViewFaceCore.Native.LibraryLoader.PathResolvers;

namespace ViewFaceCore.Native.LibraryLoader.Interface
{
    internal abstract class BaseLibraryLoader : ILibraryLoader
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
            "ViewFaceBridge"
        };

        /// <summary>
        /// 默认的路径解析器
        /// </summary>
        protected virtual IPathResolver PathResolver => new DefaultPathResolver(DeviceType.CPU);

        /// <summary>
        /// 加载静态库之前需要做的事
        /// </summary>
        protected virtual void BeforeLoad()
        {
            SetInstructionSupport();
        }

        /// <summary>
        /// 加载静态库之后需要做的事
        /// </summary>
        protected virtual void AfterLoad()
        {
            string defaultModelsPath = PathResolver.GetModelsPath();
            GlobalConfig.WriteLog($"Default models path is {defaultModelsPath}");
            SetModelsPath(defaultModelsPath);
        }

        public virtual void Load()
        {
            BeforeLoad();
            Loading();
            AfterLoad();
        }

        /// <summary>
        /// 设置模型路径
        /// </summary>
        /// <param name="path"></param>
        protected abstract void SetModelsPath(string path);

        /// <summary>
        /// 设置支持的指令集
        /// </summary>
        protected abstract void SetInstructionSupport();

        /// <summary>
        /// 加载静态库
        /// </summary>
        protected abstract void Loading();

        public abstract void Dispose();

        public virtual string GetLibraryPath()
        {
            return PathResolver.GetLibraryPath();
        }

        public virtual string GetModelsPath()
        {
            return PathResolver.GetModelsPath();
        }
    }
}
