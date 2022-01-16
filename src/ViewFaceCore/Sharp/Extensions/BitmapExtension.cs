#if !NET6_0_OR_GREATER
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using ViewFaceCore.Sharp.Model;

namespace ViewFaceCore.Extension
{
    /// <summary>
    /// <see cref="Bitmap"/> 扩展方法。
    /// </summary>
    public static class BitmapExtension
    {
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

        /// <summary>
        /// <see cref="Bitmap"/> 转换为 <see cref="FaceImage"/>。
        /// </summary>
        /// <param name="bitmap">人脸图像</param>
        /// <returns></returns>
        public static FaceImage ToFaceImage(Bitmap bitmap)
        {
            var buffer = bitmap.To24BGRByteArray(out int width, out int height, out int channels);
            return new FaceImage(width, height, channels, buffer);
        }
    }
}

#endif
