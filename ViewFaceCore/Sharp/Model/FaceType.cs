using System;

namespace ViewFaceCore.Sharp.Model
{
    /// <summary>
    /// 人脸类型
    /// </summary>
    public enum FaceType : int
    {
        /// <summary>
        /// 高精度人脸识别模型。
        /// </summary>
        Normal = 0,
        /// <summary>
        /// 戴口罩人脸识别模型。
        /// </summary>
        Mask,
        /// <summary>
        /// 轻量级人脸识别模型。
        /// </summary>
        Light,
    }
}
