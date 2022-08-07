using System;
using System.Collections.Generic;
using System.Text;

namespace ViewFaceCore.Extension.DependencyInjection
{
    /// <summary>
    /// 转入选项
    /// </summary>
    public class ViewFaceCoreOptions
    {
        /// <summary>
        /// 是否开启活体检测
        /// </summary>
        public bool IsEnableFaceAntiSpoofing { get; set; }
    }
}
