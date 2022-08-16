using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ViewFaceCore.Demo.VideoForm.Extensions
{
    public static class BitmapExtension
    {
        public static Bitmap DeepClone(this Bitmap source)
        {
            return source.Clone(new Rectangle(0, 0, source.Width, source.Height), source.PixelFormat);
        }
    }
}
