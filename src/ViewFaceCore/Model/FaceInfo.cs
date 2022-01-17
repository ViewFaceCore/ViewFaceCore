using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace ViewFaceCore.Model
{
    /// <summary>
    /// 面部信息
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct FaceInfo
    {
        private readonly FaceRect pos;
        private readonly float score;

        /// <summary>
        /// 人脸置信度
        /// </summary>
        public float Score => score;
        /// <summary>
        /// 人脸位置
        /// </summary>
        public FaceRect Location => pos;
    }
}
