using SkiaSharp;
using System.IO;
using System;

namespace ViewFaceCore.Example.WebApp.Utils
{
    public static class ImageBase64Converter
    {
        /// <summary>
        /// Converts Image to Base64
        /// </summary>
        /// <param name="image">Image</param>
        /// <returns>Base64 String</returns>
        public static string ImageToBase64(SKData image)
        {
            using (var m = new MemoryStream())
            {
                image.SaveTo(m);
                byte[] imageBytes = m.ToArray();
                string base64String = Convert.ToBase64String(imageBytes);
                return base64String;
            }
        }

        public static string ImageToBase64(SKBitmap bitmap, SKEncodedImageFormat format = SKEncodedImageFormat.Png, int quality = 100)
        {
            if (bitmap == null)
            {
                throw new ArgumentNullException(nameof(bitmap));
            }
            // create an image WRAPPER
            using (SKImage image = SKImage.FromPixels(bitmap.PeekPixels()))
            {
                // encode the image (defaults to PNG)
                using (SKData encoded = image.Encode(format, quality))
                {
                    using (var m = new MemoryStream())
                    {
                        encoded.SaveTo(m);
                        byte[] imageBytes = m.ToArray();
                        string base64String = Convert.ToBase64String(imageBytes);
                        return base64String;
                    }
                }
            }
        }

        public static string ImageToBase64WithPngFormat(SKBitmap bitmap)
        {
            return "data:image/png;base64," + ImageToBase64(bitmap, SKEncodedImageFormat.Png, 100);
        }

        /// <summary>
        /// Converts Base64-String to Image
        /// </summary>
        /// <param name="base64String"></param>
        /// <returns></returns>
        public static SKData Base64ToImage(string base64String)
        {
            if (string.IsNullOrEmpty(base64String))
            {
                throw new ArgumentNullException(base64String);
            }
            SKEncodedImageFormat format = SKEncodedImageFormat.Png;
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
                        switch (formatStr)
                        {
                            case "jpg":
                            case "jpeg":
                                format = SKEncodedImageFormat.Jpeg;
                                break;
                            case "bmp":
                                format = SKEncodedImageFormat.Bmp;
                                break;
                            case "webp":
                                format = SKEncodedImageFormat.Webp;
                                break;
                            case "png":
                            default:
                                format = SKEncodedImageFormat.Png;
                                break;
                        }
                    }
                }
                //获取图片base
                base64String = base64String.Substring(indexOfSplit + 1, base64String.Length - (indexOfSplit + 1));
            }
            byte[] imageBytes = Convert.FromBase64String(base64String);
            SKBitmap bitmap = SKBitmap.Decode(imageBytes);
            var skData = SKImage.FromBitmap(bitmap).Encode(format, 100);
            return skData;
        }
    }
}
