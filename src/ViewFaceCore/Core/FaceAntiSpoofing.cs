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
    /// 活体检测器
    /// </summary>
    public sealed class FaceAntiSpoofing : BaseViewFace, IDisposable
    {
        private readonly IntPtr _handle = IntPtr.Zero;
        private readonly static object _locker = new object();
        public FaceAntiSpoofingConfig FaceAntiSpoofingConfig { get; private set; }

        /// <summary>
        /// 活体检测器
        /// </summary>
        /// <param name="global">是否启用全局检测能力</param>
        /// <exception cref="Exception"></exception>
        public FaceAntiSpoofing(FaceAntiSpoofingConfig config = null)
        {
            this.FaceAntiSpoofingConfig = config ?? new FaceAntiSpoofingConfig();
            _handle = ViewFaceNative.GetFaceAntiSpoofingHandler(this.FaceAntiSpoofingConfig.VideoFrameCount
                , this.FaceAntiSpoofingConfig.BoxThresh
                , this.FaceAntiSpoofingConfig.Threshold.Clarity
                , this.FaceAntiSpoofingConfig.Threshold.Reality
                , this.FaceAntiSpoofingConfig.Global
                , (int)this.FaceAntiSpoofingConfig.DeviceType);
            if (_handle == IntPtr.Zero)
            {
                throw new Exception("Get face anti spoofing handler failed.");
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
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector(FaceImage)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>活体检测状态</returns>
        public AntiSpoofingResult AntiSpoofing(FaceImage image, FaceInfo info, FaceMarkPoint[] points)
        {
            lock (_locker)
            {
                float clarity = 0;
                float reality = 0;
                AntiSpoofingStatus status = (AntiSpoofingStatus)ViewFaceNative.AntiSpoofing(_handle, ref image, info.Location, points, ref clarity, ref reality);
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
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector(FaceImage)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>如果为 <see cref="AntiSpoofingStatus.Detecting"/>，则说明需要继续调用此方法，传入更多的图片</returns>
        public AntiSpoofingResult AntiSpoofingVideo(FaceImage image, FaceInfo info, FaceMarkPoint[] points)
        {
            lock (_locker)
            {
                float clarity = 0;
                float reality = 0;
                AntiSpoofingStatus status = (AntiSpoofingStatus)ViewFaceNative.AntiSpoofingVideo(_handle, ref image, info.Location, points, ref clarity, ref reality);
                return new AntiSpoofingResult(status, clarity, reality);
            }
        }

        public void Dispose()
        {
            lock (_locker)
            {
                ViewFaceNative.DisposeFaceAntiSpoofing(_handle);
            }
        }
    }
}
