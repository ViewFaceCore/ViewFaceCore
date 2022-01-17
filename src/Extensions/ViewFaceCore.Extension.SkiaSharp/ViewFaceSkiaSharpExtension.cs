using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ViewFaceCore;
using ViewFaceCore.Model;

namespace ViewFaceCore.Extension.SkiaSharp
{
    public static class ViewFaceSkiaSharpExtension
    {
        public static IEnumerable<FaceInfo> FaceDetector(this ViewFace viewFace, SKBitmap image)
        {
            byte[] data = BitmapExtension.To24BGRByteArray(image, out int width, out int height, out int channels);
            using (FaceImage faceImage = new FaceImage(width, height, channels, data))
            {
                return viewFace.FaceDetector(faceImage);
            }
        }

        public static IEnumerable<FaceMarkPoint> FaceMark(this ViewFace viewFace, SKBitmap image, FaceInfo info)
        {
            byte[] data = BitmapExtension.To24BGRByteArray(image, out int width, out int height, out int channels);
            using (FaceImage faceImage = new FaceImage(width, height, channels, data))
            {
                return viewFace.FaceMark(faceImage, info);
            }
        }
    }
}
