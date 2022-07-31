using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using ViewFaceCore.Configs;
using ViewFaceCore.Model;
using ViewFaceCore.Native;

namespace ViewFaceCore
{
    public sealed class FaceTrack : IDisposable
    {
        private IntPtr _handle = IntPtr.Zero;

        public FaceTrack(FaceTrackerConfig trackerConfig)
        {
            _handle = ViewFaceNative.GetFaceTrackHandler(trackerConfig.Width
                , trackerConfig.Height
                , trackerConfig.Type
                , trackerConfig.Stable
                , trackerConfig.Interval
                , trackerConfig.FaceSize
                , trackerConfig.Threshold);
            if (_handle == IntPtr.Zero)
            {
                throw new Exception("Get face track handler failed.");
            }
        }

        /// <summary>
        /// 识别 <paramref name="image"/> 中的人脸，并返回可跟踪的人脸信息。
        /// <para>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> <see langword="||"/> <see cref="FaceType.Light"/></c> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_detector">face_detector.csta</a><br/>
        /// 当 <c><see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/></c> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.mask_detector">mask_detector.csta</a><br/>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <returns>人脸信息集合。若 <see cref="Array.Length"/> == 0 ，代表未检测到人脸信息。如果图片中确实有人脸，可以修改 <see cref="TrackerConfig"/> 重新检测。</returns>
        public FaceTrackInfo[] Track(FaceImage image)
        {
            int size = 0;
            var ptr = ViewFaceNative.FaceTrack(_handle, ref image, ref size);
            if (ptr != IntPtr.Zero)
            {
                FaceTrackInfo[] result = new FaceTrackInfo[size];
                for (int i = 0; i < size; i++)
                {
                    int ofs = i * Marshal.SizeOf(typeof(FaceTrackInfo));
                    var info = (FaceTrackInfo)Marshal.PtrToStructure(ptr + ofs, typeof(FaceTrackInfo));
                    result[i] = info;
                }
                ViewFaceNative.Free(ptr);
                return result;
            }
            return new FaceTrackInfo[0];
        }

        /// <summary>
        /// 重置追踪的视频
        /// </summary>
        public void Reset()
        {
            ViewFaceNative.FaceTrackReset(_handle);
        }

        public void Dispose()
        {
            ViewFaceNative.FaceTrackDispose(_handle);
        }
    }
}
