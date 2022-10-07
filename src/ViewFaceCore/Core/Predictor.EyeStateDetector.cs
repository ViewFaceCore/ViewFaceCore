﻿using ViewFaceCore.Configs;
using ViewFaceCore.Exceptions;
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
    /// <exception cref="HandleInitException"></exception>
    public EyeStateDetector(EyeStateDetectConfig config = null) : base(config ?? new EyeStateDetectConfig())
    {
        if ((_handle = ViewFaceNative.GetEyeStateDetectorHandler((int)Config.DeviceType)) == IntPtr.Zero)
        {
            throw new HandleInitException("Get eye state detector handle failed.");
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
    /// <param name="points">关键点坐标<para>通过 <see cref="MaskDetector.Detect(FaceImage, FaceInfo)"/> 获取</para></param>
    /// <returns></returns>
    public EyeStateResult Detect(FaceImage image, FaceMarkPoint[] points)
    {
        lock (_locker)
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(EyeStateDetector));

            int left_eye = 0, right_eye = 0;
            ViewFaceNative.EyeStateDetect(_handle, ref image, points, ref left_eye, ref right_eye);
            return new EyeStateResult((EyeState)left_eye, (EyeState)right_eye);
        }
    }

    /// <inheritdoc/>
    public override void Dispose()
    {
        lock (_locker)
        {
            IsDisposed = true;
            ViewFaceNative.DisposeEyeStateDetector(_handle);
        }
    }
}