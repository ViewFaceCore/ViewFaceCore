using System.Collections.Generic;
using ViewFaceCore.Example.WebApp.Models.Interface;
using ViewFaceCore.Models;

namespace ViewFaceCore.Example.WebApp.Models
{
    public class PhotoDetectResponseData : IResponseData
    {
        public string Image { get; set; }

        public int Height { get; set; }

        public int Width { get; set; }

        /// <summary>
        /// 人脸信息
        /// </summary>
        public List<PhotoDetectFaceInfo> Infos { get; set; } = new List<PhotoDetectFaceInfo>();

        /// <summary>
        /// 检测耗时(ms)
        /// </summary>
        public long Elapsed { get; set; }
    }

    public class PhotoDetectFaceInfo
    {
        /// <summary>
        /// 人脸信息
        /// </summary>
        public FaceInfo FaceInfo { get; set; }

        /// <summary>
        /// 口罩检测
        /// </summary>
        public PlotMaskResult MaskResult { get; set; }

        /// <summary>
        /// 年龄
        /// </summary>
        public int? Age { get; set; }

        /// <summary>
        /// 识别
        /// </summary>
        public Gender? Gender { get; set; }

        /// <summary>
        /// 活体检测结果
        /// </summary>
        public AntiSpoofingResult AntiSpoofing { get; set; }

        /// <summary>
        /// 质量检测结果
        /// </summary>
        public QualityResult Quality { get; set; }

        /// <summary>
        /// 眼睛状态
        /// </summary>
        public EyeStateResult EyeState { get; set; }
    }

}
