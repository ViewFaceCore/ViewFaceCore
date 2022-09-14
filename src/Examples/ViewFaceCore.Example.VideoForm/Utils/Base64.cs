using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ViewFaceCore.Demo.VideoForm.Utils
{
    public static class Base64
    {
        public static string BitmapToString(Bitmap bmp)
        {
            if (bmp == null)
            {
                return null;
            }
            using (MemoryStream ms = new MemoryStream())
            {
                bmp.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
                byte[] arr = new byte[ms.Length];
                ms.Position = 0;
                ms.Read(arr, 0, (int)ms.Length);
                ms.Close();
                string strbaser64 = Convert.ToBase64String(arr);
                return strbaser64;
            }
        }


        /// <summary>
        /// base64编码的文本转为图片
        /// </summary>
        /// <param name="inputStr"></param>
        /// <returns></returns>
        public static Bitmap StringToImage(string inputStr)
        {
            if (string.IsNullOrEmpty(inputStr))
            {
                return null;
            }
            byte[] arr = Convert.FromBase64String(inputStr);
            using (MemoryStream ms = new MemoryStream(arr))
            {
                Bitmap bmp = new Bitmap(ms);
                ms.Close();
                return bmp;
            }
        }
    }
}
