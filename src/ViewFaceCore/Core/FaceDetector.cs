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

        /// <summary>
        /// 人脸检测器配置
        /// </summary>
        public FaceDetectConfig DetectConfig { get; private set; }

        /// <summary>
        /// 识别人脸的信息。
        /// </summary>
        /// <param name="config"></param>
        /// <exception cref="Exception"></exception>
        public FaceDetector(FaceDetectConfig config = null)
        {
            this.DetectConfig = config ?? new FaceDetectConfig();
            _handle = ViewFaceNative.GetFaceDetectorHandler(this.DetectConfig.FaceSize
                , this.DetectConfig.Threshold
                , this.DetectConfig.MaxWidth
                , this.DetectConfig.MaxHeight
                , (int)this.DetectConfig.DeviceType);
            if (_handle == IntPtr.Zero)
            {
                throw new Exception("Get face detector handler failed.");
            }
        }

        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回人脸的信息。
        /// <para>
        /// 可以通过 <see cref="DetectConfig"/> 属性对人脸检测器进行配置，以应对不同场景的图片。
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="DetectConfig"/> 重新检测。</returns>
        public FaceInfo[] Detect(FaceImage image)
        {
            lock (_locker)
            {
                int size = 0;
                var ptr = ViewFaceNative.FaceDetector(_handle, ref image, ref size);
                if (ptr != IntPtr.Zero)
                {
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
            return new FaceInfo[0];
        }


        /// <summary>
        /// <see cref="IDisposable"/>
        /// </summary>
        public void Dispose()
        {
            lock (_locker)
            {
                ViewFaceNative.DisposeFaceDetector(_handle);
            }
        }
    }
}
