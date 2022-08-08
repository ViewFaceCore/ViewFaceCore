using System;
using System.Collections.Generic;
using System.Text;
using ViewFaceCore.Configs;

namespace ViewFaceCore.Extension.DependencyInjection
{
    /// <summary>
    /// 转入选项
    /// </summary>
    public class ViewFaceCoreOptions
    {
        internal static string OptionName = "ViewFaceCoreOptions";

        #region 基础模块

        public FaceDetectConfig FaceDetectConfig { get; set; }

        public FaceLandmarkConfig FaceLandmarkConfig { get; set; }

        public FaceRecognizeConfig FaceRecognizeConfig { get; set; }


        #endregion

        /// <summary>
        /// 是否开启所有模块（不建议）
        /// </summary>
        public bool IsEnableAll { get; set; } = false;

        /// <summary>
        /// 是否开启活体检测
        /// </summary>
        public bool IsEnableFaceAntiSpoofing { get; set; } = false;

        public FaceAntiSpoofingConfig FaceAntiSpoofingConfig { get; set; } = null;

        /// <summary>
        /// 是否开启年龄预测
        /// </summary>
        public bool IsEnableAgePredict { get; set; } = false;

        public AgePredictConfig AgePredictConfig { get; set; } = null;

        /// <summary>
        /// 是否启用眼睛状态检测
        /// </summary>
        public bool IsEnableEyeStateDetect { get; set; } = false;

        public EyeStateDetectConfig EyeStateDetectConfig { get; set; } = null;

        /// <summary>
        /// 是否启用性别检测
        /// </summary>
        public bool IsEnableGenderPredict { get; set; } = false;

        public GenderPredictConfig GenderPredictConfig { get; set; } = null;

        /// <summary>
        /// 是否启用人脸追踪
        /// </summary>
        public bool IsEnableFaceTrack { get; set; } = false;

        public FaceTrackerConfig FaceTrackerConfig { get; set; } = new FaceTrackerConfig(3000, 3000);

        /// <summary>
        /// 是否启用口罩识别
        /// </summary>
        public bool IsEnablMaskDetect { get; set; } = false;

        public MaskDetectConfig MaskDetectConfig { get; set; } = null;

        /// <summary>
        /// 是否启用质量检测
        /// </summary>
        public bool IsEnableQuality { get; set; } = false;

        public QualityConfig QualityConfig { get; set; } = null;

    }
}
