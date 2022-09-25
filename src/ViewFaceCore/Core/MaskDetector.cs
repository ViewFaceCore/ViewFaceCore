using ViewFaceCore.Configs;
using ViewFaceCore.Exceptions;
using ViewFaceCore.Models;
using ViewFaceCore.Native;

namespace ViewFaceCore.Core;

/// <summary>
/// 口罩人脸识别
/// </summary>
public sealed class MaskDetector : BaseViewFace<MaskDetectConfig>
{
    private readonly IntPtr _handle = IntPtr.Zero;
    private readonly static object _locker = new object();

    /// <summary>
    /// 口罩人脸识别
    /// </summary>
    /// <param name="config"></param>
    /// <exception cref="HandleInitException"></exception>
    public MaskDetector(MaskDetectConfig config = null) : base(config ?? new MaskDetectConfig())
    {
        if ((_handle = ViewFaceNative.GetMaskDetectorHandler((int)Config.DeviceType)) == IntPtr.Zero)
        {
            throw new HandleInitException("Get mask detector handle failed.");
        }
    }

    /// <summary>
    /// 戴口罩人脸识别
    /// </summary>
    /// <param name="image"></param>
    /// <param name="info"></param>
    /// <returns></returns>
    public PlotMaskResult Detect(FaceImage image, FaceInfo info)
    {
        lock (_locker)
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(MaskDetector));

            float score = 0;
            bool status = ViewFaceNative.MaskDetect(_handle, ref image, info.Location, ref score);
            return new PlotMaskResult(score, status, status && score > this.Config.Threshold);
        }
    }

    /// <inheritdoc/>
    public override void Dispose()
    {
        lock (_locker)
        {
            IsDisposed = true;
            ViewFaceNative.DisposeMaskDetector(_handle);
        }
    }
}
