using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using ViewFaceCore.Model;

namespace ViewFaceCore
{
    public static class ViewFaceSystemDrawingExtension
    {
        //public static IEnumerable<FaceInfo> FaceDetector(this ViewFace viewFace, Bitmap image)
        //{
        //    byte[] data = BitmapExtension.To24BGRByteArray(image, out int width, out int height, out int channels);
        //    IEnumerable<FaceInfo> result = null;
        //    FaceImage faceImage = new FaceImage(width, height, channels, data);
        //    result = viewFace.FaceDetector(faceImage);
        //    return result;
        //}

        //public static IEnumerable<FaceMarkPoint> FaceMark(this ViewFace viewFace, Bitmap image, FaceInfo info)
        //{
        //    byte[] data = BitmapExtension.To24BGRByteArray(image, out int width, out int height, out int channels);
        //    using (FaceImage faceImage = new FaceImage(width, height, channels, data))
        //    {
        //        return viewFace.FaceMark(faceImage, info);
        //    }
        //}

        public static FaceImage ToFaceImage(this Bitmap image)
        {
            byte[] data = To24BGRByteArray(image, out int width, out int height, out int channels);
            FaceImage faceImage = new FaceImage(width, height, channels, data);
            return faceImage;
        }
        public static FaceImage ToFaceImage(this object obj)
        {
            if (obj is Bitmap bitmap)
            {
                return bitmap.ToFaceImage();
            }
            throw new NotImplementedException();
        }

        #region private

        /// <summary>
        /// <see cref="Bitmap"/> 转为 3*8bit BGR <see cref="byte"/> 数组。
        /// </summary>
        /// <param name="bitmap">待转换图像</param>
        /// <param name="width">图像宽度</param>
        /// <param name="height">图像高度</param>
        /// <param name="channels">图像通道</param>
        /// <returns>图像的 BGR <see cref="byte"/> 数组</returns>
        public static byte[] To24BGRByteArray(this Bitmap bitmap, out int width, out int height, out int channels)
        {
            width = bitmap.Width;
            height = bitmap.Height;
            channels = 3;
            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppRgb);
            try
            {
                int num = bitmap.Height * bitmapData.Stride;
                byte[] array = new byte[num];
                Marshal.Copy(bitmapData.Scan0, array, 0, num);
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
            finally
            {
                bitmap.UnlockBits(bitmapData);
            }
        }

        #endregion
    }
}
