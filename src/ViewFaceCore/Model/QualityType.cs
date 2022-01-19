using System.ComponentModel;

namespace ViewFaceCore.Model
{
    /// <summary>
    /// 质量评估类型
    /// </summary>
    [Description("质量评估类型")]
    public enum QualityType
    {
        /// <summary>
        /// 亮度评估
        /// <para>亮度评估就是评估人脸区域内的亮度值是否均匀正常，存在部分或全部的过亮和过暗都会是评价为LOW。</para>
        /// </summary>
        [Description("亮度评估")]
        Brightness = 0,
        /// <summary>
        /// 清晰度评估
        /// <para>清晰度这里是传统方式通过二次模糊后图像信息损失程度统计的清晰度。</para>
        /// </summary>
        [Description("清晰度评估")]
        Clarity,
        /// <summary>
        /// 完整度评估
        /// <para>完整度评估是朴素的判断人脸是否因为未完全进入摄像头而造成的不完整的情况。该方法不适用于判断遮挡造成的不完整。</para>
        /// </summary>
        [Description("完整度评估")]
        Integrity,
        /// <summary>
        /// 姿态评估
        /// <para>此姿态评估器是传统方式，通过人脸5点坐标值来判断姿态是否为正面。</para>
        /// </summary>
        [Description("姿态评估")]
        Pose,
        /// <summary>
        /// 姿态评估（深度）
        /// <para>此姿态评估器是深度学习方式，通过回归人头部在yaw、pitch、roll三个方向的偏转角度来评估人脸是否是正面。</para>
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.pose_estimation">pose_estimation.csta</a> 
        /// </para>
        /// </summary>
        [Description("姿态评估（深度）")]
        PoseEx,
        /// <summary>
        /// 分辨率评估
        /// <para>判断人脸部分的分辨率。</para>
        /// </summary>
        [Description("分辨率评估")]
        Resolution,
        /// <summary>
        /// 清晰度评估（深度）
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.quality_lbn">quality_lbn.csta</a> <br />
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_pts68">face_landmarker_pts68.csta</a> 
        /// </para>
        /// </summary>
        [Description("清晰度评估（深度）")]
        ClarityEx,
        /// <summary>
        /// 遮挡评估
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_mask_pts5">face_landmarker_mask_pts5.csta</a> 
        /// </para>
        /// </summary>
        [Description("遮挡评估")]
        Structure,
    }
}
