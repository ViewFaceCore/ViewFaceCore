using System;
using System.Collections.Generic;
using System.Text;

namespace ViewFaceCore.Native.LibraryLoader.Interface
{
    public interface IPathResolver
    {
        /// <summary>
        /// 获取静态库默认路径
        /// </summary>
        /// <returns></returns>
        string GetLibraryPath();

        /// <summary>
        /// 获取默认模型文件路径
        /// </summary>
        /// <returns></returns>
        string GetModelsPath();

        /// <summary>
        /// 获取库完整路径
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        string GetLibraryFullName(string name);
    }
}
