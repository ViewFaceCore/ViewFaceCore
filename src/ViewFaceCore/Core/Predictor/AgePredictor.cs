using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ViewFaceCore.Configs;
using ViewFaceCore.Models;
using ViewFaceCore.Native;

namespace ViewFaceCore.Core;

/// <summary>
/// 年龄预测，需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.age_predictor">age_predictor.csta</a>
/// </summary>
public sealed class AgePredictor : BaseViewFace<AgePredictConfig>, IPredictor
{
    private readonly IntPtr _handle = IntPtr.Zero;
    private readonly static object _locker = new object();

    /// <inheritdoc/>
    /// <exception cref="Exception"></exception>
    public AgePredictor(AgePredictConfig config = null) : base(config ?? new AgePredictConfig())
    {
        if ((_handle = ViewFaceNative.GetAgePredictorHandler((int)Config.DeviceType)) == IntPtr.Zero)
        {
            throw new Exception("Get age predictor handler failed.");
        }
    }

    /// <summary>
    /// 年龄预测。
    /// <para>
    /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.age_predictor">age_predictor.csta</a>
    /// </para>
    /// </summary>
    /// <param name="image">人脸图像信息</param>
    /// <param name="points">关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
    /// <returns>-1: 预测失败失败，其它: 预测的年龄。</returns>
    public int PredictAge(FaceImage image, FaceMarkPoint[] points)
    {
        lock (_locker)
        {
            return ViewFaceNative.PredictAge(_handle, ref image, points);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_locker)
        {
            ViewFaceNative.DisposeAgePredictor(_handle);
        }
    }
}
