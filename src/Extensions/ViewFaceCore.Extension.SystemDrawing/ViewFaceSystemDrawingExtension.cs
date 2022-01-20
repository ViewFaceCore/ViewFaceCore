using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using ViewFaceCore.Model;

namespace ViewFaceCore
{
    public static class ViewFaceSystemDrawingExtension
    {
        /// <summary>
        /// Bitmap convert to FaceImage
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        public static FaceImage ToFaceImage(this Bitmap image)
        {
            byte[] data = To24BGRByteArray(image, out int width, out int height, out int channels);
            FaceImage faceImage = new FaceImage(width, height, channels, data);
            return faceImage;
        }

        /// <summary>
        /// Bitmap convert to FaceImage
        /// </summary>
        /// <typeparam name="T">Only support type of System.Drawing.Bitmap</typeparam>
        /// <param name="obj"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public static FaceImage ToFaceImage<T>(this T obj) where T : class
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
        private static byte[] To24BGRByteArray(this Bitmap bitmap, out int width, out int height, out int channels)
        {
            if (bitmap == null)
            {
                throw new ArgumentNullException(nameof(bitmap));
            }
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
