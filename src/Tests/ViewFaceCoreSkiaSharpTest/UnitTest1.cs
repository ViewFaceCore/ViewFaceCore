using Microsoft.VisualStudio.TestTools.UnitTesting;
using SkiaSharp;
using System.Linq;
using ViewFaceCore;

namespace ViewFaceCoreSkiaSharpTest
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestMethod1()
        {
            using SKBitmap bitmap = SKBitmap.Decode(SKData.Create(imagePath));
            ViewFace viewFace = new ViewFace();
            for (int i = 0; i < 1000; i++)
            {
                var infos = viewFace.FaceDetector(bitmap);
                System.Console.WriteLine(string.Join(System.Environment.NewLine, infos.Select(x => $"{i}. {x.Score} - {{{x.Location.X},{x.Location.Y}}} - {{{x.Location.Width},{x.Location.Height}}}")));
            }
        }

        private readonly static string imagePath = @"images\Jay_3.jpg";

        [TestMethod]
        public void FaceDetectorTest()
        {
            using SKBitmap bitmap = SKBitmap.Decode(SKData.Create(imagePath));
            ViewFace viewFace = new ViewFace();
            var infos = viewFace.FaceDetector(bitmap);
            Assert.IsTrue(infos.Any() && infos.First().Score > 0 && infos.First().Location.X > 0 && infos.First().Location.Y > 0 && infos.First().Location.Width > 0 && infos.First().Location.Height > 0);
        }

        [TestMethod]
        public void FaceMarkTest()
        {
            using SKBitmap bitmap = SKBitmap.Decode(SKData.Create(imagePath));
            ViewFace viewFace = new ViewFace();
            var infos = viewFace.FaceDetector(bitmap);
            var info = infos.First();
            var points = viewFace.FaceMark(bitmap, info).ToList();
            System.Console.WriteLine(points.Count);
            Assert.IsTrue(points.Any());
        }
    }
}