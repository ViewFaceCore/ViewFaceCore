using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace ViewFaceCore.Sharp.Extensions
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
            Bitmap bmp;
            Rectangle rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            // Get the 24bppRgb Bitmap
            if (bitmap.PixelFormat != PixelFormat.Format24bppRgb)
            {
                bmp = new Bitmap(bitmap.Width, bitmap.Height, PixelFormat.Format24bppRgb);
                Graphics.FromImage(bmp).DrawImageEx(bitmap, rect, rect, GraphicsUnit.Pixel).Dispose();
            }
            else { bmp = (Bitmap)bitmap.Clone(); }

            BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            width = bmpData.Width;
            height = bmpData.Height;
            channels = bmpData.Stride / bmpData.Width;
            int bytesLength = bmpData.Height * bmpData.Stride; // Get the BGR values's length.
            byte[] buffer = new byte[bytesLength]; // Create the BGR buffer.
            Marshal.Copy(bmpData.Scan0, buffer, 0, bytesLength); // Copy the BGR values into the array.
            bmp.UnlockBits(bmpData);
            bmp.Dispose();
            return buffer;
        }
    }
}
