using System.Collections.Generic;
using System.Linq;
using ViewFaceCore.Sharp.Model;

namespace ViewFaceCore.Sharp.Extensions
{
    /// <summary>
    /// <see cref="ViewFace"/> 的扩展方法类
    /// </summary>
    public static class ViewFaceExtension
    {
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
        public static AntiSpoofingStatus AntiSpoofingVideo(this ViewFace viewFace, IEnumerable<FaceImage> bitmaps, int faceIndex, bool global)
        {
            var result = AntiSpoofingStatus.Detecting;
            bool haveFace = false;
            foreach (var bitmap in bitmaps)
            {
                var infos = viewFace.FaceDetector(bitmap).ToArray();
                if (faceIndex >= 0 && faceIndex < infos.Length)
                {
                    haveFace = true;
                    var points = viewFace.FaceMark(bitmap, infos[faceIndex]);
                    var status = viewFace.AntiSpoofingVideo(bitmap, infos[faceIndex], points, global);
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
    }
}
