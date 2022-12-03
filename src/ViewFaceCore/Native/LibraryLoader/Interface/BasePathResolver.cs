using ViewFaceCore.Configs.Enums;

namespace ViewFaceCore.Native.LibraryLoader.Interface
{
    public abstract class BasePathResolver : IPathResolver
    {
        protected DeviceType DeviceType { get; private set; }

        public BasePathResolver(DeviceType deviceType)
        {
            this.DeviceType = deviceType;
        }

        public abstract string GetModelsPath();

        public abstract string GetLibraryPath();

        public abstract string GetLibraryFullName(string name);
    }
}
