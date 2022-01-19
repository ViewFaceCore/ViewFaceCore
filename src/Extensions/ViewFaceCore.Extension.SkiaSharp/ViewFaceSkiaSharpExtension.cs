using SkiaSharp;
using System;
using ViewFaceCore.Model;

namespace ViewFaceCore
{
    public static class ViewFaceSkiaSharpExtension
    {
        //public static IEnumerable<FaceInfo> FaceDetector(this ViewFace viewFace, SKBitmap image)
        //{
        //    byte[] data = BitmapExtension.To24BGRByteArray(image, out int width, out int height, out int channels);
        //    FaceImage faceImage = new FaceImage(width, height, channels, data);
        //    return viewFace.FaceDetector(faceImage);
        //}

        //public static IEnumerable<FaceMarkPoint> FaceMark(this ViewFace viewFace, SKBitmap image, FaceInfo info)
        //{
        //    byte[] data = BitmapExtension.To24BGRByteArray(image, out int width, out int height, out int channels);
        //    using (FaceImage faceImage = new FaceImage(width, height, channels, data))
        //    {
        //        return viewFace.FaceMark(faceImage, info);
        //    }
        //}

        public static FaceImage ToFaceImage(this SKBitmap image)
        {
            byte[] data = To24BGRByteArray(image, out int width, out int height, out int channels);
            FaceImage faceImage = new FaceImage(width, height, channels, data);
            return faceImage;
        }
        public static FaceImage ToFaceImage(this object obj)
        {
            if (obj is SKBitmap bitmap)
            {
                return bitmap.ToFaceImage();
            }
            throw new NotImplementedException();
        }

        #region Private

        /// <summary>
        /// <see cref="Bitmap"/> 转为 3*8bit BGR <see cref="byte"/> 数组。
        /// </summary>
        /// <param name="bitmap">待转换图像</param>
        /// <param name="width">图像宽度</param>
        /// <param name="height">图像高度</param>
        /// <param name="channels">图像通道</param>
        /// <returns>图像的 BGR <see cref="byte"/> 数组</returns>
        private static byte[] To24BGRByteArray(this SKBitmap bitmap, out int width, out int height, out int channels)
        {
            width = bitmap.Width;
            height = bitmap.Height;
            channels = 3;
            byte[] array = bitmap.Bytes;
            byte[] bgra = new byte[array.Length / 4 * channels];
            // brga
            int j = 0;
            for (int i = 0; i < array.Length; i++)
            {
                if ((i + 1) % 4 == 0) continue;
                bgra[j] = array[i];
                j++;
            }
            return bgra;
        }

        #endregion
    }
}
