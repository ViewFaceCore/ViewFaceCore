using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using ViewFaceCore.Configs;
using ViewFaceCore.Model;
using ViewFaceCore.Native;

namespace ViewFaceCore.Core
{
    /// <summary>
    /// 识别人脸的信息。
    /// </summary>
    public sealed class FaceDetector : BaseViewFace, IDisposable
    {
        private readonly IntPtr _handle = IntPtr.Zero;
        private readonly static object _locker = new object();
        public FaceDetectorConfig DetectorConfig { get; private set; }

        public FaceDetector(FaceDetectorConfig detectorConfig = null)
        {
            this.DetectorConfig = detectorConfig ?? new FaceDetectorConfig();
            _handle = ViewFaceNative.GetFaceDetectorHandler(this.DetectorConfig.FaceSize
                , this.DetectorConfig.Threshold
                , this.DetectorConfig.MaxWidth
                , this.DetectorConfig.MaxHeight);
            if (_handle == IntPtr.Zero)
            {
                throw new Exception("Get face detector handler failed.");
            }
        }

        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回人脸的信息。
        /// <para>
        /// 可以通过 <see cref="DetectorConfig"/> 属性对人脸检测器进行配置，以应对不同场景的图片。
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="DetectorConfig"/> 重新检测。</returns>
        public FaceInfo[] Detect(FaceImage image)
        {
            lock (_locker)
            {
                int size = 0;
                var ptr = ViewFaceNative.FaceDetector(_handle, ref image, ref size);
                if (ptr != IntPtr.Zero)
                {
                    FaceInfo[] result = new FaceInfo[size];
                    for (int i = 0; i < size; i++)
                    {
                        int ofs = i * Marshal.SizeOf(typeof(FaceInfo));
                        result[i] = (FaceInfo)Marshal.PtrToStructure(ptr + ofs, typeof(FaceInfo));
                    }
                    ViewFaceNative.Free(ptr);
                    return result;
                }
            }
            return new FaceInfo[0];
        }

        public void Dispose()
        {
            lock (_locker)
            {
                ViewFaceNative.DisposeFaceDetector(_handle);
            }
        }
    }
}
