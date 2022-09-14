using ViewFaceCore.Configs.Enums;

namespace ViewFaceCore.Configs;

/// <summary>
/// 人脸关键的检测配置
/// </summary>
public sealed class FaceLandmarkConfig : BaseConfig
{
    /// <summary>
    /// 关键点类型 
    /// <para>
    /// 当关键点类型为 <see cref="MarkType.Light"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_pts5">face_landmarker_pts5.csta</a><br/>
    /// 当关键点类型为 <see cref="MarkType.Normal"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_pts68">face_landmarker_pts68.csta</a><br/>
    /// 当关键点类型为 <see cref="MarkType.Mask"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_mask_pts5">face_landmarker_mask_pts5.csta</a><br/>
    /// </para>
    /// </summary>
    /// <remarks>
    /// 默认值: <see cref="MarkType.Light"/>
    /// </remarks>
    public MarkType MarkType { get; set; } = MarkType.Light;
}
