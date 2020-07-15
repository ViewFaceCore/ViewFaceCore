using System;
using System.Collections.Generic;

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

    /// <summary>
    /// 人脸相似度阈值
    /// </summary>
    public class Face
    {
        /// <summary>
        /// 人脸相似度阈值
        /// </summary>
        public static Dictionary<FaceType, float> Threshold { get; } = new Dictionary<FaceType, float>()
        {
            { FaceType.Normal,0.62f },
            { FaceType.Mask,0.48f },
            { FaceType.Light,0.55f },
        };
    }
}
