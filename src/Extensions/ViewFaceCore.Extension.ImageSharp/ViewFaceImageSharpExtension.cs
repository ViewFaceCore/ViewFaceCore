using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ViewFaceCore.Model;

namespace ViewFaceCore.Extension.ImageSharp
{
    public static class ViewFaceImageSharpExtension
    {
        ///// <summary>
        ///// SKBitmap convert to FaceImage
        ///// </summary>
        ///// <param name="image"></param>
        ///// <returns></returns>
        //public static FaceImage ToFaceImage(this Image image)
        //{
        //    byte[] data = To24BGRByteArray(image, out int width, out int height, out int channels);
        //    FaceImage faceImage = new FaceImage(width, height, channels, data);
        //    return faceImage;
        //}

        ///// <summary>
        ///// <see cref="Bitmap"/> 转为 3*8bit BGR <see cref="byte"/> 数组。
        ///// </summary>
        ///// <param name="source">待转换图像</param>
        ///// <param name="width">图像宽度</param>
        ///// <param name="height">图像高度</param>
        ///// <param name="channels">图像通道</param>
        ///// <returns>图像的 BGR <see cref="byte"/> 数组</returns>
        //private static byte[] To24BGRByteArray(Image source, out int width, out int height, out int channels)
        //{
        //    if (source == null)
        //    {
        //        throw new ArgumentNullException(nameof(source));
        //    }
        //    source.PixelType

        //    channels = 3;
        //    if (source.ColorType != targetColorType)
        //    {
        //        using (SKBitmap bitmap = ConvertToBgra8888(source))
        //        {
        //            width = bitmap.Width;
        //            height = bitmap.Height;
        //            return ConvertToBGRByte(bitmap, channels);
        //        }
        //    }
        //    else
        //    {
        //        width = source.Width;
        //        height = source.Height;
        //        return ConvertToBGRByte(source, channels);
        //    }
        //}
    }
}
