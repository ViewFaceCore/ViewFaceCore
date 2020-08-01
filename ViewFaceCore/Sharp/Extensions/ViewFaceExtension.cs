using System.Drawing;

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
        /// <param name="bitmaps">一组图片，即视频帧的 <see cref="Bitmap"/> 数组</param>
        /// <param name="faceIndex">指定要识别的人脸索引</param>
        /// <param name="global">是否启用全局检测能力</param>
        /// <returns></returns>
        public static AntiSpoofingStatus AntiSpoofingVideo(this ViewFace viewFace, Bitmap[] bitmaps, int faceIndex, bool global)
        {
            var result = AntiSpoofingStatus.Detecting;
            bool haveFace = false;
            foreach (var bitmap in bitmaps)
            {
                var infos = viewFace.FaceDetector(bitmap);
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

        /// <summary>
        /// 计算人脸特征值相似度。
        /// </summary>
        /// <param name="viewFace"></param>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns></returns>
        public static float Compare(this ViewFace viewFace, float[] lhs, float[] rhs)
        {
            float sum = 0;
            for (int i = 0; i < lhs.Length; i++)
            {
                sum += lhs[i] * rhs[i];
            }
            return sum;
        }
    }
}
