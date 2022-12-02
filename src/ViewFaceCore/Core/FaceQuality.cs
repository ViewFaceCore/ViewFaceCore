using ViewFaceCore.Configs;
using ViewFaceCore.Configs.Enums;
using ViewFaceCore.Exceptions;
using ViewFaceCore.Models;
using ViewFaceCore.Native;

namespace ViewFaceCore.Core;

/// <summary>
/// 质量评估
/// </summary>
public sealed class FaceQuality : BaseViewFace<QualityConfig>
{
    private readonly IntPtr _clarityHandle = IntPtr.Zero;
    private readonly static object _clarityLocker = new object();

    private readonly IntPtr _maskHandle = IntPtr.Zero;
    private readonly static object _maskLocker = new object();

    /// <inheritdoc/>
    public FaceQuality(QualityConfig config = null) : base(config ?? new QualityConfig())
    {
        _clarityHandle = ViewFaceNative.GetQualityOfClarityExHandler(this.Config.ClarityEx.BlurThresh, (int)this.Config.DeviceType);
        if (_clarityHandle == IntPtr.Zero)
        {
            throw new ModuleInitializeException("Get quality of clarityEx handle failed.");
        }

        _maskHandle = ViewFaceNative.GetQualityOfNoMaskHandler((int)this.Config.DeviceType);
        if (_maskHandle == IntPtr.Zero)
        {
            throw new ModuleInitializeException("Get quality of nomask handle failed.");
        }
    }

    /// <summary>
    /// 人脸质量评估
    /// </summary>
    /// <param name="image">人脸图像信息</param>
    /// <param name="info">面部信息<para>通过 <see cref="FaceDetector.Detect(FaceImage)"/> 获取</para></param>
    /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceLandmarker.Mark(FaceImage, FaceInfo)"/> 获取</para></param>
    /// <param name="type">质量评估类型</param>
    /// <returns></returns>
    public QualityResult Detect(FaceImage image, FaceInfo info, FaceMarkPoint[] points, QualityType type)
    {
        int level = -1;
        float score = -1;

        switch (type)
        {
            case QualityType.Brightness:
                ViewFaceNative.QualityOfBrightness(ref image, info.Location, points, points.Length, ref level, ref score,
                    this.Config.Brightness.V0, this.Config.Brightness.V1, this.Config.Brightness.V2, this.Config.Brightness.V3);
                break;
            case QualityType.Clarity:
                ViewFaceNative.QualityOfClarity(ref image, info.Location, points, points.Length, ref level, ref score, this.Config.Clarity.Low, this.Config.Clarity.High);
                break;
            case QualityType.Integrity:
                ViewFaceNative.QualityOfIntegrity(ref image, info.Location, points, points.Length, ref level, ref score,
                    this.Config.Integrity.Low, this.Config.Integrity.High);
                break;
            case QualityType.Pose:
                ViewFaceNative.QualityOfPose(ref image, info.Location, points, points.Length, ref level, ref score);
                break;
            case QualityType.PoseEx:
                ViewFaceNative.QualityOfPoseEx(ref image, info.Location, points, points.Length, ref level, ref score,
                   this.Config.PoseEx.YawLow, this.Config.PoseEx.YawHigh,
                   this.Config.PoseEx.PitchLow, this.Config.PoseEx.PitchHigh,
                   this.Config.PoseEx.RollLow, this.Config.PoseEx.RollHigh);
                break;
            case QualityType.Resolution:
                ViewFaceNative.QualityOfResolution(ref image, info.Location, points, points.Length, ref level, ref score, this.Config.Resolution.Low, this.Config.Resolution.High);
                break;
            case QualityType.ClarityEx:
                {
                    if (IsDisposed)
                        throw new ObjectDisposedException(nameof(QualityType.ClarityEx));
                    lock (_clarityLocker)
                    {
                        ViewFaceNative.QualityOfClarityEx(_clarityHandle, ref image, info.Location, points, points.Length, ref level, ref score);
                    }
                }
                break;
            case QualityType.Structure:
                {
                    if (IsDisposed)
                        throw new ObjectDisposedException(nameof(QualityType.Structure));
                    lock (_maskLocker)
                    {
                        ViewFaceNative.QualityOfNoMask(_maskHandle, ref image, info.Location, points, points.Length, ref level, ref score);
                    }
                }
                break;
        }

        return new QualityResult((QualityLevel)level, score, type);
    }

    /// <inheritdoc/>
    public override void Dispose()
    {
        IsDisposed = true;

        lock (_clarityLocker)
        {
            ViewFaceNative.DisposeQualityOfClarityEx(_clarityHandle);
        }

        lock (_maskLocker)
        {
            ViewFaceNative.DisposeQualityOfNoMask(_maskHandle);
        }
    }
}
