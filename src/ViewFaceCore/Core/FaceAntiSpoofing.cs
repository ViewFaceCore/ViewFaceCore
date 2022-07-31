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

        /// <summary>
        /// 活体检测器
        /// </summary>
        /// <param name="global">是否启用全局检测能力</param>
        /// <exception cref="Exception"></exception>
        public FaceAntiSpoofing(bool global = false)
        {
            _handle = ViewFaceNative.GetFaceAntiSpoofingHandler(global);
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
        public AntiSpoofingStatus AntiSpoofing(FaceImage image, FaceInfo info, FaceMarkPoint[] points)
        {
            lock (_locker)
            {
                return (AntiSpoofingStatus)ViewFaceNative.AntiSpoofing(_handle, ref image, info.Location, points);
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
        public AntiSpoofingStatus AntiSpoofingVideo(FaceImage image, FaceInfo info, FaceMarkPoint[] points)
        {
            lock (_locker)
            {
                return (AntiSpoofingStatus)ViewFaceNative.AntiSpoofingVideo(_handle, ref image, info.Location, points);
            }
        }

        /// <summary>
        /// 活体检测器。
        /// <para>
        /// 视频帧图片，由 <paramref name="global"/> 指定是否启用全局检测能力 <br />
        /// </para>
        /// <para>如果返回结果为 <see cref="AntiSpoofingStatus.Detecting"/>，则说明需要继续调用此方法，传入更多的图片</para>
        /// </summary>
        /// <param name="viewFace"></param>
        /// <param name="bitmaps">一组图片信息，即视频帧的 <see cref="FaceImage"/> 数组</param>
        /// <param name="faceIndex">指定要识别的人脸索引</param>
        /// <param name="global">是否启用全局检测能力</param>
        /// <returns></returns>
        public AntiSpoofingStatus AntiSpoofingVideo(IEnumerable<FaceImage> bitmaps, int faceIndex = 0)
        {
            using FaceDetector faceDetector = new FaceDetector();
            using FaceMark faceMark = new FaceMark();
            var result = AntiSpoofingStatus.Detecting;
            bool haveFace = false;
            foreach (var bitmap in bitmaps)
            {
                var infos = faceDetector.Detector(bitmap);
                if (faceIndex >= 0 && faceIndex < infos.Length)
                {
                    haveFace = true;
                    var points = faceMark.Mark(bitmap, infos[faceIndex]);
                    var status = AntiSpoofingVideo(bitmap, infos[faceIndex], points);
                    if (status == AntiSpoofingStatus.Detecting)
                    { continue; }
                    else { result = status; }
                }
            }
            if (haveFace)
            { return result; }
            else
            { return AntiSpoofingStatus.Error; }
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
