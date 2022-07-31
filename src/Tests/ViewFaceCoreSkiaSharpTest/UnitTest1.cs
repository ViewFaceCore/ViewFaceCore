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
        private readonly static string imagePath1 = @"images\Jay_4.jpg";

        [TestMethod]
        public void FaceDetectorTest1()
        {
            using SKBitmap bitmap = SKBitmap.Decode(imagePath);
            using ViewFace viewFace = new ViewFace();

            var infos = viewFace.FaceDetector(bitmap).ToList();
            Assert.IsTrue(infos.Any() && infos.First().Score > 0 && infos.First().Location.X > 0 && infos.First().Location.Y > 0 && infos.First().Location.Width > 0 && infos.First().Location.Height > 0);

            var infos1 = viewFace.FaceDetector(bitmap).ToArray();
        }

        [TestMethod]
        public void FaceMarkTest1()
        {
            using SKBitmap bitmap = SKBitmap.Decode(imagePath);
            using ViewFace viewFace = new ViewFace();
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
        public void FaceTrackTest1()
        {
            using SKBitmap bitmap = SKBitmap.Decode(imagePath);

            using (FaceTrack faceTrack = new FaceTrack(new ViewFaceCore.Configs.FaceTrackerConfig(bitmap.Width, bitmap.Height)))
            {
                var result = faceTrack.Track(bitmap).ToList();
                if (result == null || !result.Any())
                {
                    Assert.Fail();
                }
                faceTrack.Reset();
            }

            Debug.WriteLine("FaceTrack is disposed");
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

        [TestMethod]
        public void CompareTest()
        {
            using SKBitmap bitmap0 = SKBitmap.Decode(imagePath);
            using SKBitmap bitmap1 = SKBitmap.Decode(imagePath1);
            using ViewFace viewFace = new ViewFace();

            var p0 = GetExtract(viewFace, bitmap0);
            var p1 = GetExtract(viewFace, bitmap1);

            float result = viewFace.Compare(p0, p1);
            bool isSelf = viewFace.IsSelf(p0, p1);
            Assert.IsTrue(isSelf);
        }


        #region Helpers

        private IEnumerable<FaceMarkPoint> GetFaceMarkPoint(ViewFace viewFace, SKBitmap bitmap)
        {
            var infos = viewFace.FaceDetector(bitmap);
            var info = infos.First();
            return viewFace.FaceMark(bitmap, info).ToList();
        }

        private float[] GetExtract(ViewFace viewFace, SKBitmap bitmap)
        {
            return viewFace.Extract(bitmap, GetFaceMarkPoint(viewFace, bitmap));
        }

        #endregion
    }
}