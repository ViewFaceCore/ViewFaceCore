using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using ViewFaceCore.Configs;
using ViewFaceCore.Exceptions;
using ViewFaceCore.Models;
using ViewFaceCore.Native;

namespace ViewFaceCore.Core;

/// <summary>
/// 识别人脸的信息。
/// </summary>
public sealed class FaceDetector : BaseViewFace<FaceDetectConfig>, IDisposable
{
    private readonly IntPtr _handle = IntPtr.Zero;
    private readonly static object _locker = new object();

    /// <inheritdoc/>
    /// <exception cref="HandleInitException"></exception>
    public FaceDetector(FaceDetectConfig config = null) : base(config ?? new FaceDetectConfig())
    {
        _handle = ViewFaceNative.GetFaceDetectorHandler(this.Config.FaceSize, this.Config.Threshold, this.Config.MaxWidth, this.Config.MaxHeight, (int)this.Config.DeviceType);
        if (_handle == IntPtr.Zero)
        {
            throw new HandleInitException("Get face detector handle failed.");
        }
    }

    /// <summary>
    /// 识别 <paramref name="image"/> 中的人脸，并返回人脸的信息。
    /// <para>
    /// 可以通过 <see cref="FaceDetectConfig.FaceDetectConfig"/> 属性对人脸检测器进行配置，以应对不同场景的图片。
    /// </para>
    /// </summary>
    /// <param name="image">人脸图像信息</param>
    /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="Config"/> 重新检测。</returns>
    public FaceInfo[] Detect(FaceImage image)
    {
        lock (_locker)
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(FaceAntiSpoofing));

            int size = 0;
            var ptr = ViewFaceNative.FaceDetector(_handle, ref image, ref size);
            if (ptr == IntPtr.Zero) return new FaceInfo[0];
            try
            {
                FaceInfo[] result = new FaceInfo[size];
                for (int i = 0; i < size; i++)
                {
                    int ofs = i * Marshal.SizeOf(typeof(FaceInfo));
                    result[i] = (FaceInfo)Marshal.PtrToStructure(ptr + ofs, typeof(FaceInfo));
                }
                return result.OrderBy(p => p.Score).ToArray();
            }
            finally
            {
                ViewFaceNative.Free(ptr);
            }
        }
    }

    /// <summary>
    /// <see cref="IDisposable"/>
    /// </summary>
    public override void Dispose()
    {
        lock (_locker)
        {
            IsDisposed = true;
            ViewFaceNative.DisposeFaceDetector(_handle);
        }
    }
}
