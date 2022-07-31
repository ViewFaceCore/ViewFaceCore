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
    /// 识别指定的人脸信息的关键点坐标。
    /// </summary>
    public sealed class FaceMark : BaseViewFace, IDisposable
    {
        private readonly IntPtr _handle = IntPtr.Zero;
        private readonly static object _locker = new object();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="type">模型类型。0：face_landmarker_pts68；1：face_landmarker_mask_pts5；2：face_landmarker_pts5。</param>
        /// <exception cref="Exception"></exception>
        public FaceMark(MarkType markType = MarkType.Light)
        {
            _handle = ViewFaceNative.GetFaceLandmarkerHandler((int)markType);
            if (_handle == IntPtr.Zero)
            {
                throw new Exception("Get face landmarker handler failed.");
            }
        }

        /// <summary>
        /// 识别 <paramref name="image"/> 中指定的人脸信息 <paramref name="info"/> 的关键点坐标。
        /// <para>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Normal"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_pts68">face_landmarker_pts68.csta</a><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Mask"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_mask_pts5">face_landmarker_mask_pts5.csta</a><br/>
        /// 当 <see cref="FaceType"/> <see langword="="/> <see cref="FaceType.Light"/> 时， 需要模型：<a href="https://www.nuget.org/packages/ViewFaceCore.model.face_landmarker_pts5">face_landmarker_pts5.csta</a><br/>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">指定的人脸信息</param>
        /// <exception cref="MarkException"/>
        /// <returns>若失败，则返回结果 Length == 0</returns>
        public FaceMarkPoint[] Mark(FaceImage image, FaceInfo info)
        {
            lock (_locker)
            {
                long size = 0;
                var ptr = ViewFaceNative.FaceMark(_handle, ref image, info.Location, ref size);
                if (ptr != IntPtr.Zero)
                {
                    FaceMarkPoint[] result = new FaceMarkPoint[size];
                    for (int i = 0; i < size; i++)
                    {
                        var ofs = i * Marshal.SizeOf(typeof(FaceMarkPoint));
                        result[i] = (FaceMarkPoint)Marshal.PtrToStructure(ptr + ofs, typeof(FaceMarkPoint));
                    }
                    ViewFaceNative.Free(ptr);
                }
            }
            return new FaceMarkPoint[0];
        }

        public void Dispose()
        {
            lock (_locker)
            {
                ViewFaceNative.DisposeFaceLandmarker(_handle);
            }
        }
    }
}
