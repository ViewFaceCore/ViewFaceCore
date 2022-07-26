using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Diagnostics;
using System.Linq;
using ViewFaceCore;

namespace ViewFaceCoreTest
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void FaceDetectorTest()
        {
            using System.Drawing.Bitmap bitmap = (System.Drawing.Bitmap)System.Drawing.Image.FromFile(@"images\Jay_3.jpg");
            ViewFace viewFace = new ViewFace();

            for (int i = 0; i < 1000; i++)
            {
                var infos = viewFace.FaceDetector(bitmap);
                Debug.WriteLine(string.Join(System.Environment.NewLine, infos.Select(x => $"{i}. {x.Score} - {{{x.Location.X},{x.Location.Y}}} - {{{x.Location.Width},{x.Location.Height}}}")));
            }
        }
    }
}