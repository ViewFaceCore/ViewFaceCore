using ViewFaceCore.Configs;
using ViewFaceCore.Exceptions;
using ViewFaceCore.Models;
using ViewFaceCore.Native;

namespace ViewFaceCore.Core;

/// <summary>
/// 性别预测。
/// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.gender_predictor">gender_predictor.csta</a>
/// </summary>
public sealed class GenderPredictor : Predictor<GenderPredictConfig>
{
    private readonly IntPtr _handle = IntPtr.Zero;
    private readonly static object _locker = new object();

    /// <inheritdoc/>
    /// <exception cref="ModuleInitializeException"></exception>
    public GenderPredictor(GenderPredictConfig config = null) : base(config ?? new GenderPredictConfig())
    {
        if ((_handle = ViewFaceNative.GetGenderPredictorHandler((int)Config.DeviceType)) == IntPtr.Zero)
        {
            throw new ModuleInitializeException("Get gender predictor handle failed.");
        }
    }

    /// <summary>
    /// 性别预测
    /// <para>
    /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.gender_predictor">gender_predictor.csta</a>
    /// </para>
    /// </summary>
    /// <param name="image">人脸图像信息</param>
    /// <returns>性别。<see cref="Gender.Unknown"/> 代表识别失败</returns>
    public Gender PredictGender(FaceImage image)
    {
        lock (_locker)
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(GenderPredictor));

            int result = ViewFaceNative.PredictGender(_handle, ref image);
            if (Enum.TryParse(result.ToString(), out Gender gender))
            {
                return gender;
            }
            return Gender.Unknown;
        }
    }

    /// <summary>
    /// 性别预测（自动裁剪）
    /// <para>
    /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.gender_predictor">gender_predictor.csta</a>
    /// </para>
    /// </summary>
    /// <param name="image">人脸图像信息</param>
    /// <param name="points">关键点坐标<para>通过 <see cref="MaskDetector.Detect(FaceImage, FaceInfo)"/> 获取</para></param>
    /// <returns>性别。<see cref="Gender.Unknown"/> 代表识别失败</returns>
    public Gender PredictGenderWithCrop(FaceImage image, FaceMarkPoint[] points)
    {
        lock (_locker)
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(GenderPredictor));

            int result = ViewFaceNative.PredictGenderWithCrop(_handle, ref image, points);
            if (Enum.TryParse(result.ToString(), out Gender gender))
            {
                return gender;
            }
            return Gender.Unknown;
        }
    }

    /// <inheritdoc/>
    public override void Dispose()
    {
        lock (_locker)
        {
            IsDisposed = true;
            ViewFaceNative.DisposeGenderPredictor(_handle);
        }
    }
}
