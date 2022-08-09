using SkiaSharp;
using System;
using ViewFaceCore.Model;

namespace ViewFaceCore
{
    public static class ViewFaceSkiaSharpExtension
    {
        private const SKColorType targetColorType = SKColorType.Bgra8888;

        /// <summary>
        /// SKBitmap convert to FaceImage
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        public static FaceImage ToFaceImage(this SKBitmap image)
        {
            byte[] data = To24BGRByteArray(image, out int width, out int height, out int channels);
            return new FaceImage(width, height, channels, data);
        }

        /// <summary>
        /// SKBitmap convert to FaceImage
        /// </summary>
        /// <typeparam name="T">Only support type of SkiaSharp.SKBitmap</typeparam>
        /// <param name="obj"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public static FaceImage ToFaceImage<T>(this T obj) where T : class
        {
            if (obj == null)
            {
                throw new ArgumentNullException(nameof(obj));
            }
            if (obj is SKBitmap bitmap)
            {
                return bitmap.ToFaceImage();
            }
            throw new Exception($"Not support type:{obj.GetType()}");
        }

        #region Private

        /// <summary>
        /// <see cref="Bitmap"/> 转为 3*8bit BGR <see cref="byte"/> 数组。
        /// </summary>
        /// <param name="source">待转换图像</param>
        /// <param name="width">图像宽度</param>
        /// <param name="height">图像高度</param>
        /// <param name="channels">图像通道</param>
        /// <returns>图像的 BGR <see cref="byte"/> 数组</returns>
        private static byte[] To24BGRByteArray(SKBitmap source, out int width, out int height, out int channels)
        {
            if (source == null)
            {
                throw new ArgumentNullException(nameof(source));
            }
            channels = 3;
            if (source.ColorType != targetColorType)
            {
                using (SKBitmap bitmap = ConvertToBgra8888(source))
                {
                    width = bitmap.Width;
                    height = bitmap.Height;
                    return ConvertToByte(bitmap, channels);
                }
            }
            else
            {
                width = source.Width;
                height = source.Height;
                return ConvertToByte(source, channels);
            }
        }

        /// <summary>
        /// 转换图像格式
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        private static SKBitmap ConvertToBgra8888(SKBitmap source)
        {
            if (!source.CanCopyTo(targetColorType))
            {
                throw new Exception("Can not copy image color type to Bgra8888");
            }
            SKBitmap bitmap = new SKBitmap();
            source.CopyTo(bitmap, targetColorType);
            if (bitmap == null)
            {
                throw new Exception("Copy image to Bgra8888 failed");
            }
            return bitmap;
        }

        /// <summary>
        /// 转为BGR Bytes
        /// </summary>
        /// <param name="source"></param>
        /// <param name="channels"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        private static byte[] ConvertToByte(SKBitmap source, int channels)
        {
            byte[] array = source.Bytes;
            if (array == null || array.Length == 0)
            {
                throw new Exception("SKBitmap data is null");
            }
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
