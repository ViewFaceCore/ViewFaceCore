using Microsoft.VisualStudio.TestTools.UnitTesting;

using System;
using System.Diagnostics;
using System.Drawing;

using ViewFaceCore.Sharp;
using ViewFaceCore.Sharp.Model;

namespace ViewFaceCore.MSTest
{
    [TestClass]
    public class UnitTest1
    {
        static string[] images = new string[]
        {
            "Images\\Jay0.jpg",
            "Images\\Jay1.jpg",
            "Images\\Jay2.jpg",
            "Images\\Jay3.jpg",
        };

        [TestMethod]
        [DataRow(false)]
        public void TestAntiSpoofing(bool global)
        {
            if (!(Image.FromFile(images[0]) is Bitmap bitmap)) { throw new ArgumentNullException(nameof(bitmap)); }

            ViewFace viewFace = new ViewFace((str) => { Console.WriteLine(str); });
            var infos = viewFace.FaceDetector(bitmap);
            if (infos.Length > 0)
            {
                var points = viewFace.FaceMark(bitmap, infos[0]);
                AntiSpoofingStatus status = viewFace.AntiSpoofing(bitmap, infos[0], points, global);
                Console.WriteLine(status);
            }
            else
            { Console.WriteLine("No Face!"); }

            bitmap.Dispose();
        }

        [TestMethod]
        [DataRow(true)]
        public void TestAntiSpoofingVideo(bool global)
        {
            if (!(Image.FromFile(images[0]) is Bitmap bitmap)) { throw new ArgumentNullException(nameof(bitmap)); }

            ViewFace viewFace = new ViewFace((str) => { System.Console.WriteLine(str); });
            var infos = viewFace.FaceDetector(bitmap);
            if (infos.Length > 0)
            {
                var points = viewFace.FaceMark(bitmap, infos[0]);
                AntiSpoofingStatus status = AntiSpoofingStatus.Error;
                while (true)
                {
                    status = viewFace.AntiSpoofingVideo(bitmap, infos[0], points, global);
                    if (status != AntiSpoofingStatus.Detecting)
                    { break; }
                }
                Console.WriteLine(status);
            }
            else
            { Console.WriteLine("No Face!"); }

            bitmap.Dispose();
        }
    }
}
