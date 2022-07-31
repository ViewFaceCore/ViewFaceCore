using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Drawing;
using System.Linq;
using ViewFaceCore;
using ViewFaceCore.Core.Interface;

namespace ViewFaceCoreTest
{
    [TestClass]
    public class ViewFaceBaseTest
    {
        private readonly static string imagePath = @"images\Jay_3.jpg";

        [TestMethod]
        public void FaceDetectorTest()
        {
            using Bitmap bitmap = (Bitmap)Image.FromFile(imagePath);
            BaseViewFace viewFace = new BaseViewFace();
            var infos = viewFace.FaceDetector(bitmap);
            Assert.IsTrue(infos.Any() && infos.First().Score > 0 && infos.First().Location.X > 0 && infos.First().Location.Y > 0 && infos.First().Location.Width > 0 && infos.First().Location.Height > 0);
        }

        [TestMethod]
        public void FaceMarkTest()
        {
            using Bitmap bitmap = (Bitmap)Image.FromFile(imagePath);
            BaseViewFace viewFace = new BaseViewFace();
            var infos = viewFace.FaceDetector(bitmap);
            var info = infos.First();
            var points = viewFace.FaceMark(bitmap, info).ToList();
            System.Console.WriteLine(points.Count);
            Assert.IsTrue(points.Any());
        }

    }
}