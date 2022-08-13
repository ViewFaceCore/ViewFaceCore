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
    /// 口罩人脸识别
    /// </summary>
    public sealed class MaskDetector : BaseViewFace, IDisposable
    {
        private readonly IntPtr _handle = IntPtr.Zero;
        private readonly static object _locker = new object();
        public MaskDetectConfig DetectConfig { get; private set; }

        /// <summary>
        /// 口罩人脸识别
        /// </summary>
        /// <param name="config"></param>
        /// <exception cref="Exception"></exception>
        public MaskDetector(MaskDetectConfig config = null)
        {
            this.DetectConfig = config ?? new MaskDetectConfig();
            _handle = ViewFaceNative.GetMaskDetectorHandler((int)this.DetectConfig.DeviceType);
            if (_handle == IntPtr.Zero)
            {
                throw new Exception("Get mask detector handler failed.");
            }
        }

        /// <summary>
        /// 戴口罩人脸识别
        /// </summary>
        /// <param name="image"></param>
        /// <param name="info"></param>
        /// <param name="score">一般性的，score超过0.5，则认为是检测带上了口罩</param>
        /// <returns></returns>
        public PlotMaskResult PlotMask(FaceImage image, FaceInfo info)
        {
            lock (_locker)
            {
                float score = 0;
                bool status = ViewFaceNative.PlotMask(_handle, ref image, info.Location, ref score);
                return new PlotMaskResult(score, status, status && score > this.DetectConfig.Threshold);
            }
        }

        /// <summary>
        /// 释放人脸识别对象
        /// </summary>
        public void Dispose()
        {
            lock (_locker)
            {
                ViewFaceNative.DisposeMaskDetector(_handle);
            }
        }
    }
}
