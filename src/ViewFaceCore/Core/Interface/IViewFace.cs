using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ViewFaceCore.Core
{
    public interface IViewFace
    {
        /// <summary>
        /// 获取模型路径
        /// </summary>
        public string ModelPath { get; }

        /// <summary>
        /// 获取库路径
        /// </summary>
        public string LibraryPath { get; }
    }
}
