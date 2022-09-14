using System;
using ViewFaceCore.Configs;
using ViewFaceCore.Models;
using ViewFaceCore.Native;

namespace ViewFaceCore.Core;

/// <summary>
/// 口罩人脸识别
/// </summary>
public sealed class MaskDetector : BaseViewFace<MaskDetectConfig>, IDisposable
{
    private readonly IntPtr _handle = IntPtr.Zero;
    private readonly static object _locker = new object();

    /// <summary>
    /// 口罩人脸识别
    /// </summary>
    /// <param name="config"></param>
    /// <exception cref="Exception"></exception>
    public MaskDetector(MaskDetectConfig config = null) : base(config ?? new MaskDetectConfig())
    {
        if ((_handle = ViewFaceNative.GetMaskDetectorHandler((int)Config.DeviceType)) == IntPtr.Zero)
        {
            throw new Exception("Get mask detector handler failed.");
        }
    }

    /// <summary>
    /// 戴口罩人脸识别
    /// </summary>
    /// <param name="image"></param>
    /// <param name="info"></param>
    /// <returns></returns>
    public PlotMaskResult PlotMask(FaceImage image, FaceInfo info)
    {
        lock (_locker)
        {
            float score = 0;
            bool status = ViewFaceNative.PlotMask(_handle, ref image, info.Location, ref score);
            return new PlotMaskResult(score, status, status && score > this.Config.Threshold);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_locker)
        {
            ViewFaceNative.DisposeMaskDetector(_handle);
        }
    }
}
