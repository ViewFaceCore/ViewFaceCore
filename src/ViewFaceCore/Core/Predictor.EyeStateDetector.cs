using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ViewFaceCore.Configs;
using ViewFaceCore.Models;
using ViewFaceCore.Native;

namespace ViewFaceCore.Core;

/// <summary>
/// 眼睛状态检测。<br />
/// 眼睛的左右是相对图片内容而言的左右。<br />
/// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.eye_state">eye_state.csta</a>
/// </summary>
public sealed class EyeStateDetector : Predictor<EyeStateDetectConfig>
{
    private readonly IntPtr _handle = IntPtr.Zero;
    private readonly static object _locker = new object();

    /// <inheritdoc/>
    /// <exception cref="Exception"></exception>
    public EyeStateDetector(EyeStateDetectConfig config = null) : base(config ?? new EyeStateDetectConfig())
    {
        if ((_handle = ViewFaceNative.GetEyeStateDetectorHandler((int)Config.DeviceType)) == IntPtr.Zero)
        {
            throw new Exception("Get eye state detector handler failed.");
        }
    }

    /// <summary>
    /// 眼睛状态检测。
    /// <para>
    /// 眼睛的左右是相对图片内容而言的左右。<br />
    /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.eye_state">eye_state.csta</a>
    /// </para>
    /// </summary>
    /// <param name="image">人脸图像信息</param>
    /// <param name="points">关键点坐标<para>通过 <see cref="MaskDetector.PlotMask(FaceImage, FaceInfo)"/> 获取</para></param>
    /// <returns></returns>
    public EyeStateResult Detect(FaceImage image, FaceMarkPoint[] points)
    {
        lock (_locker)
        {
            int left_eye = 0, right_eye = 0;
            ViewFaceNative.EyeStateDetector(_handle, ref image, points, ref left_eye, ref right_eye);
            return new EyeStateResult((EyeState)left_eye, (EyeState)right_eye);
        }
    }

    /// <inheritdoc/>
    public override void Dispose()
    {
        lock (_locker)
        {
            ViewFaceNative.DisposeEyeStateDetector(_handle);
        }
    }
}
