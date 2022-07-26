using Microsoft.VisualStudio.TestTools.UnitTesting;
using SkiaSharp;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using ViewFaceCore;
using ViewFaceCore.Model;

namespace ViewFaceCoreSkiaSharpTest
{
    [TestClass]
    public class UnitTest1
    {
        private readonly static string imagePath = @"images\Jay_3.jpg";

        [TestMethod]
        public void FaceDetectorTest1()
        {
            using SKBitmap bitmap = SKBitmap.Decode(imagePath);
            ViewFace viewFace = new ViewFace();
            var infos = viewFace.FaceDetector(bitmap);
            Assert.IsTrue(infos.Any() && infos.First().Score > 0 && infos.First().Location.X > 0 && infos.First().Location.Y > 0 && infos.First().Location.Width > 0 && infos.First().Location.Height > 0);
        }

        [TestMethod]
        public void FaceMarkTest1()
        {
            using SKBitmap bitmap = SKBitmap.Decode(imagePath);
            ViewFace viewFace = new ViewFace();
            var infos = viewFace.FaceDetector(bitmap);
            var info = infos.First();
            var points = viewFace.FaceMark(bitmap, info).ToList();
            System.Console.WriteLine(points.Count);
            Assert.IsTrue(points.Any());
        }

        [TestMethod]
        public void ExtractTest1()
        {
            using SKBitmap bitmap = SKBitmap.Decode(imagePath);
            ViewFace viewFace = new ViewFace();
            var result = viewFace.Extract(bitmap, GetFaceMarkPoint(viewFace, bitmap));
            Assert.IsTrue(result.Any());
        }

        [TestMethod]
        public void AntiSpoofingTest1()
        {
            using SKBitmap bitmap = SKBitmap.Decode(imagePath);
            ViewFace viewFace = new ViewFace();
            var infos = viewFace.FaceDetector(bitmap);
            var info = infos.First();
            var markPoints = GetFaceMarkPoint(viewFace, bitmap);
            var result = viewFace.AntiSpoofing(bitmap, info, markPoints);
            Assert.IsTrue(result == AntiSpoofingStatus.Real);
        }

        [TestMethod]
        public void AntiSpoofingTest2()
        {
            using SKBitmap bitmap = SKBitmap.Decode(imagePath);
            ViewFace viewFace = new ViewFace();
            var infos = viewFace.FaceDetector(bitmap);
            var info = infos.First();
            var markPoints = GetFaceMarkPoint(viewFace, bitmap);
            for (int i = 0; i < 1000; i++)
            {
                var result = viewFace.AntiSpoofing(bitmap, info, markPoints);
                Assert.IsTrue(result == AntiSpoofingStatus.Real);
                Debug.WriteLine($"第{i + 1}次检测，正常！");
            }
        }


        private IEnumerable<FaceMarkPoint> GetFaceMarkPoint(ViewFace viewFace, SKBitmap bitmap)
        {
            var infos = viewFace.FaceDetector(bitmap);
            var info = infos.First();
            return viewFace.FaceMark(bitmap, info).ToList();
        }
    }
}