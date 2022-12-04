using ViewFaceCore.Configs.Enums;

namespace ViewFaceCore.Native.LibraryLoader.Interface
{
    /// <summary>
    /// 路径解析器基类
    /// </summary>
    public abstract class BasePathResolver : IPathResolver
    {
        /// <summary>
        /// 设备类型
        /// </summary>
        protected DeviceType DeviceType { get; private set; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="deviceType"></param>
        public BasePathResolver(DeviceType deviceType)
        {
            this.DeviceType = deviceType;
        }

        /// <summary>
        /// 获取默认模型文件路径
        /// </summary>
        /// <returns></returns>
        public abstract string GetModelsPath();

        /// <summary>
        /// 获取静态库默认路径
        /// </summary>
        /// <returns></returns>
        public abstract string GetLibraryPath();

        /// <summary>
        /// 获取库完整路径
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public virtual string GetLibraryFullName(string name)
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
                throw new PlatformNotSupportedException($"Unsupported operating system type: {RuntimeInformation.OSDescription}");
            return Path.Combine(GetLibraryPath(), string.Format(format, name));
        }
    }
}
