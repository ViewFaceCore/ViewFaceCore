using SkiaSharp;
using System;
using System.IO;

namespace ViewFaceCore.Example.WebApp.Utils
{
    public static class ImageBase64Converter
    {
        /// <summary>
        /// Converts Image to Base64
        /// </summary>
        /// <param name="image">Image</param>
        /// <returns>Base64 String</returns>
        public static string ImageToBase64(SKData image, out SKEncodedImageFormat format)
        {
            format = SKEncodedImageFormat.Png;
            if (image == null)
            {
                return null;
            }
            using (SKCodec codec = SKCodec.Create(image))
            {
                format = codec.EncodedFormat;
                using (var m = new MemoryStream())
                {
                    image.SaveTo(m);
                    byte[] imageBytes = m.ToArray();
                    string base64String = Convert.ToBase64String(imageBytes);
                    return base64String;
                }
            }
        }

        /// <summary>
        /// Converts SKBitmap to Base64 with format
        /// </summary>
        /// <param name="image"></param>
        /// <param name="format"></param>
        /// <returns></returns>
        public static string ImageToBase64WithFormat(SKData image, out SKEncodedImageFormat format)
        {
            format = SKEncodedImageFormat.Png;
            if (image == null)
            {
                return null;
            }
            string base64Row = ImageToBase64(image, out format);
            return $"data:image/{format.ToString().ToLower()};base64,{base64Row}";
        }

        /// <summary>
        /// Converts SKBitmap to Base64 with PNG format
        /// </summary>
        /// <param name="bitmap"></param>
        /// <returns></returns>
        public static string BitmapToBase64WithFormat(SKBitmap bitmap, SKEncodedImageFormat format)
        {
            return $"data:image/{format.ToString().ToLower()};base64,{BitmapToBase64(bitmap, format)}";
        }

        /// <summary>
        /// Converts SKBitmap to Base64
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        public static string BitmapToBase64(SKBitmap bitmap, SKEncodedImageFormat format)
        {
            if (bitmap == null)
            {
                throw new ArgumentNullException(nameof(bitmap));
            }
            // create an image WRAPPER
            using (SKImage image = SKImage.FromBitmap(bitmap))
            {
                //不支持bmp编码
                if (format == SKEncodedImageFormat.Bmp)
                {
                    throw new NotSupportedException($"Not support encode {format}.");
                }
                // encode the image (defaults to PNG)
                using (SKData encoded = image.Encode(format, 100))
                {
                    return ImageToBase64(encoded, out _);
                }
            }
        }

        /// <summary>
        /// Converts Base64-String to SKBitmap
        /// </summary>
        /// <param name="base64String"></param>
        /// <returns></returns>
        public static SKBitmap Base64ToBitmap(string base64String, out SKEncodedImageFormat format)
        {
            if (string.IsNullOrEmpty(base64String))
            {
                throw new ArgumentNullException(base64String);
            }
            format = SKEncodedImageFormat.Png;
            int indexOfSplit = base64String.LastIndexOf(',');
            if (indexOfSplit != -1)
            {
                //获取图片格式
                int index1 = base64String.IndexOf(':') + 1;
                int index2 = base64String.IndexOf(';');
                if (index2 - index1 > 0)
                {
                    string formatStr = base64String.Substring(index1, index2 - index1);
                    if (!string.IsNullOrEmpty(formatStr) && formatStr.StartsWith("image", StringComparison.OrdinalIgnoreCase))
                    {
                        formatStr = formatStr.ToLower().Replace("image/", "");
                        if (!System.Enum.TryParse(formatStr, true, out format))
                        {
                            format = SKEncodedImageFormat.Png;
                        }
                    }
                }
                //获取图片base
                base64String = base64String.Substring(indexOfSplit + 1, base64String.Length - (indexOfSplit + 1));
            }
            byte[] imageBytes = Convert.FromBase64String(base64String);
            SKBitmap bitmap = SKBitmap.Decode(imageBytes);
            return bitmap;
        }

        /// <summary>
        /// Converts Base64-String to Image
        /// </summary>
        /// <param name="base64String"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentNullException"></exception>
        public static SKData Base64ToImage(string base64String, out SKEncodedImageFormat format)
        {
            if (string.IsNullOrEmpty(base64String))
            {
                throw new ArgumentNullException(base64String);
            }
            using (SKBitmap bitmap = Base64ToBitmap(base64String, out format))
            {
                //不支持bmp编码
                if (format == SKEncodedImageFormat.Bmp)
                {
                    format = SKEncodedImageFormat.Png;
                }
                var skData = SKImage.FromBitmap(bitmap).Encode(format, 100);
                return skData;
            }
        }
    }
}
