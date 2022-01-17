using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using ViewFaceCore;
using ViewFaceCore.Model;

namespace ViewFaceCore.Extension.SystemDrawing
{
    public static class ViewFaceSystemDrawingExtension
    {
        public static IEnumerable<FaceInfo> FaceDetector(this ViewFace viewFace, Bitmap image)
        {
            byte[] data = BitmapExtension.To24BGRByteArray(image, out int width, out int height, out int channels);
            using (FaceImage faceImage = new FaceImage(width, height, channels, data))
            {
                return viewFace.FaceDetector(faceImage);
            }
        }

        public static IEnumerable<FaceMarkPoint> FaceMark(this ViewFace viewFace, Bitmap image, FaceInfo info)
        {
            byte[] data = BitmapExtension.To24BGRByteArray(image, out int width, out int height, out int channels);
            using (FaceImage faceImage = new FaceImage(width, height, channels, data))
            {
                return viewFace.FaceMark(faceImage, info);
            }
        }
    }
}
