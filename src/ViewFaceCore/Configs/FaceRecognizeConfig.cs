using System.Collections.Generic;
using ViewFaceCore.Model;

namespace ViewFaceCore.Configs
{
    /// <summary>
    /// 人脸对比阈值配置
    /// </summary>
    public class FaceRecognizeConfig : BaseConfig
    {
        public FaceRecognizeConfig()
        {
            FaceType = FaceType.Normal;
        }

        public FaceRecognizeConfig(FaceType faceType)
        {
            FaceType = faceType;
        }

        public FaceType FaceType { get; set; } = FaceType.Normal;

        /// <summary>
        /// 人脸相似度阈值
        /// </summary>
        private static Dictionary<FaceType, float> Threshold { get; } = new Dictionary<FaceType, float>()
        {
            { FaceType.Normal, 0.62f },
            { FaceType.Mask, 0.48f },
            { FaceType.Light, 0.55f },
        };

        /// <summary>
        /// 获取指定人脸识别模型的相似度阈值。
        /// <para>
        /// • <see cref="FaceType.Normal"/> 的默认相似度阈值为 <see langword="0.62"/> <br />
        /// • <see cref="FaceType.Mask"/> 的默认相似度阈值为 <see langword="0.48"/> <br />
        /// • <see cref="FaceType.Light"/> 的默认相似度阈值为 <see langword="0.55"/> <br />
        /// </para>
        /// </summary>
        /// <param name="type">指定的人脸识别模型</param>
        /// <returns></returns>
        public static float GetThreshold(FaceType type) => Threshold[type];
        /// <summary>
        /// 设置指定人脸识别模型的相似度阈值。
        /// <para>
        /// 默认阈值是一般场景使用的推荐阈值。一般来说1比1的场景下，该阈值会对应偏低，1比N场景会对应偏高。
        /// </para>
        /// <para>
        /// • <see cref="FaceType.Normal"/> 的默认相似度阈值为 <see langword="0.62"/> <br />
        /// • <see cref="FaceType.Mask"/> 的默认相似度阈值为 <see langword="0.48"/> <br />
        /// • <see cref="FaceType.Light"/> 的默认相似度阈值为 <see langword="0.55"/> <br />
        /// </para>
        /// </summary>
        /// <param name="type">指定的人脸识别模型</param>
        /// <param name="score">相似度阈值</param>
        public static void SetThreshold(FaceType type, float score) => Threshold[type] = score;
    }
}
