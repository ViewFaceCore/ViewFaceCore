using ViewFaceCore.Configs;
using ViewFaceCore.Exceptions;
using ViewFaceCore.Models;
using ViewFaceCore.Native;

namespace ViewFaceCore.Core;

/// <summary>
/// 活体检测器
/// </summary>
public sealed class FaceAntiSpoofing : BaseViewFace<FaceAntiSpoofingConfig>
{
    private readonly IntPtr _handle = IntPtr.Zero;
    private readonly static object _locker = new object();

    /// <inheritdoc/>
    /// <exception cref="HandleInitException"></exception>
    public FaceAntiSpoofing(FaceAntiSpoofingConfig config = null) : base(config ?? new FaceAntiSpoofingConfig())
    {
        _handle = ViewFaceNative.GetFaceAntiSpoofingHandler(Config.VideoFrameCount, Config.BoxThresh, Config.Threshold.Clarity, Config.Threshold.Reality, Config.Global
            , (int)this.Config.DeviceType);
        if (_handle == IntPtr.Zero)
        {
            throw new HandleInitException("Get face anti spoofing handle failed.");
        }
    }

    /// <summary>
    /// 活体检测器。(单帧图片)
    /// <para>
    /// 当 <paramref name="global"/> <see langword="= false"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.fas_first">fas_first.csta</a><br/>
    /// 当 <paramref name="global"/> <see langword="= true"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.fas_second">fas_second.csta</a>
    /// </para>
    /// </summary>
    /// <param name="image">人脸图像信息</param>
    /// <param name="info">面部信息<para>通过 <see cref="FaceDetector.Detect(FaceImage)"/> 获取</para></param>
    /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="MaskDetector.PlotMask(FaceImage, FaceInfo)"/> 获取</para></param>
    /// <returns>活体检测状态</returns>
    public AntiSpoofingResult Predict(FaceImage image, FaceInfo info, FaceMarkPoint[] points)
    {
        lock (_locker)
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(FaceAntiSpoofing));

            float clarity = 0;
            float reality = 0;
            AntiSpoofingStatus status = (AntiSpoofingStatus)ViewFaceNative.FaceAntiSpoofingPredict(_handle, ref image, info.Location, points, ref clarity, ref reality);
            return new AntiSpoofingResult(status, clarity, reality);
        }
    }

    /// <summary>
    /// 活体检测器。(视频帧图片)
    /// <para>
    /// 当 <paramref name="global"/> <see langword="= false"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.fas_first">fas_first.csta</a><br/>
    /// 当 <paramref name="global"/> <see langword="= true"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.fas_second">fas_second.csta</a>
    /// </para>
    /// </summary>
    /// <param name="image">人脸图像信息</param>
    /// <param name="info">面部信息<para>通过 <see cref="FaceDetector.Detect(FaceImage)"/> 获取</para></param>
    /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceLandmarker.Mark(FaceImage, FaceInfo)"/> 获取</para></param>
    /// <returns>如果为 <see cref="AntiSpoofingStatus.Detecting"/>，则说明需要继续调用此方法，传入更多的图片</returns>
    public AntiSpoofingResult PredictVideo(FaceImage image, FaceInfo info, FaceMarkPoint[] points)
    {
        lock (_locker)
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(FaceAntiSpoofing));

            float clarity = 0, reality = 0;
            AntiSpoofingStatus status = (AntiSpoofingStatus)ViewFaceNative.FaceAntiSpoofingPredictVideo(_handle, ref image, info.Location, points, ref clarity, ref reality);
            return new AntiSpoofingResult(status, clarity, reality);
        }
    }

    /// <summary>
    /// 释放非托管资源
    /// </summary>
    public override void Dispose()
    {
        lock (_locker)
        {
            IsDisposed = true;
            ViewFaceNative.DisposeFaceAntiSpoofing(_handle);
        }
    }
}
