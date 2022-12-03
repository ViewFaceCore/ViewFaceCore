using System;
using System.Collections.Generic;
using System.Text;

namespace ViewFaceCore.Native.LibraryLoader.Interface
{
    /// <summary>
    /// 静态库加载器
    /// </summary>
    internal interface ILibraryLoader : IDisposable
    {
        void SetPathResolver(IPathResolver pathResolver);

        /// <summary>
        /// 加载静态库
        /// </summary>
        void Load();

        /// <summary>
        /// 获取设置的静态库路径
        /// </summary>
        /// <returns></returns>
        string GetLibraryPath();

        /// <summary>
        /// 获取设置的模型库路径
        /// </summary>
        /// <returns></returns>
        string GetModelsPath();
    }
}
