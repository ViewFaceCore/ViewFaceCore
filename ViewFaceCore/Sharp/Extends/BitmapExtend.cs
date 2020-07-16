using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace ViewFaceCore.Sharp.Extends
{
    static class BitmapExtend
    {
        /// <summary>
        /// <see cref="Bitmap"/> 转为 3*8bit BGR <see cref="byte"/> 数组。
        /// </summary>
        /// <param name="bitmap">待转换图像</param>
        /// <param name="width">图像宽度</param>
        /// <param name="height">图像高度</param>
        /// <param name="channels">图像扫描宽度</param>
        /// <returns></returns>
        public static byte[] To24BGRByteArray(this Bitmap bitmap, out int width, out int height, out int channels)
        {
            Bitmap bmp = (Bitmap)bitmap.Clone();
            if (bitmap.PixelFormat != PixelFormat.Format24bppRgb)
            {
                bmp = new Bitmap(bitmap.Width, bitmap.Height, PixelFormat.Format24bppRgb);
                using (Graphics g = Graphics.FromImage(bmp))
                {
                    g.DrawImage(bitmap, new Rectangle(0, 0, bitmap.Width, bitmap.Height));
                }
            }
            if (bmp.HorizontalResolution != 96 || bmp.VerticalResolution != 96)
            { bmp.SetResolution(96, 96); }

            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            int bytesLength = bmp.Width * bmp.Height * 3; // Get the BGR values's length.
            width = bmp.Width;
            height = bmp.Height;
            channels = 3;
            byte[] buffer = new byte[bytesLength];
            // Copy the BGR values into the array.
            Marshal.Copy(bmpData.Scan0, buffer, 0, bytesLength);
            bmp.UnlockBits(bmpData);
            bmp.Dispose();
            return buffer;
        }
    }
}
