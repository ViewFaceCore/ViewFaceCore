using ViewFaceCore.Configs;
using ViewFaceCore.Exceptions;
using ViewFaceCore.Models;
using ViewFaceCore.Native;

namespace ViewFaceCore.Core;

/// <summary>
/// 年龄预测，需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.age_predictor">age_predictor.csta</a>
/// </summary>
public sealed class AgePredictor : Predictor<AgePredictConfig>
{
    private readonly IntPtr _handle = IntPtr.Zero;
    private readonly static object _locker = new object();

    /// <inheritdoc/>
    /// <exception cref="HandleInitException"></exception>
    public AgePredictor(AgePredictConfig config = null) : base(config ?? new AgePredictConfig())
    {
        if ((_handle = ViewFaceNative.GetAgePredictorHandler((int)Config.DeviceType)) == IntPtr.Zero)
        {
            throw new HandleInitException("Get age predictor handle failed.");
        }
    }

    /// <summary>
    /// 年龄预测
    /// <para>
    /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.age_predictor">age_predictor.csta</a>
    /// </para>
    /// </summary>
    /// <param name="image">人脸图像信息</param>
    /// <returns>-1: 预测失败失败，其它: 预测的年龄。</returns>
    public int PredictAge(FaceImage image)
    {
        lock (_locker)
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(AgePredictor));

            return ViewFaceNative.PredictAge(_handle, ref image);
        }
    }

    /// <summary>
    /// 年龄预测（自动裁剪）
    /// <para>
    /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.age_predictor">age_predictor.csta</a>
    /// </para>
    /// </summary>
    /// <param name="image">人脸图像信息</param>
    /// <param name="points">关键点坐标<para>通过 <see cref="MaskDetector.PlotMask(FaceImage, FaceInfo)"/> 获取</para></param>
    /// <returns>-1: 预测失败失败，其它: 预测的年龄。</returns>
    public int PredictAgeWithCrop(FaceImage image, FaceMarkPoint[] points)
    {
        lock (_locker)
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(AgePredictor));

            return ViewFaceNative.PredictAgeWithCrop(_handle, ref image, points);
        }
    }

    /// <inheritdoc/>
    public override void Dispose()
    {
        lock (_locker)
        {
            IsDisposed = true;
            ViewFaceNative.DisposeAgePredictor(_handle);
        }
    }
}
