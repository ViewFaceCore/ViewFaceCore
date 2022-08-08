using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Diagnostics;
using System.Linq;
using ViewFaceCore.Core;
using ViewFaceCore.Model;
using ViewFaceCore.Extension.ImageSharp;
using SixLabors.ImageSharp;

namespace ViewFaceCoreImageSharpTest
{
    [TestClass]
    public class UnitTest1
    {
        private readonly static string imagePath = @"images/Jay_3.jpg";
        private readonly static string imagePath1 = @"images/Jay_4.jpg";
        private readonly static string maskImagePath = @"images/mask_01.jpeg";

        [TestMethod]
        public void FaceDetectorAndFaceMarkTest()
        {
            using var bitmap = ConvertImage(imagePath);
            //using FaceDetector faceDetector = new FaceDetector();
            //using FaceLandmarker faceMark = new FaceLandmarker();

            //Stopwatch sw = Stopwatch.StartNew();

            //var infos = faceDetector.Detect(bitmap);
            //var info = infos.First();
            //var markPoints = GetFaceMarkPoint(faceDetector, faceMark, bitmap);

            //sw.Stop();
            //Debug.WriteLine($"{nameof(FaceLandmarker.Mark)}识别，结果：{markPoints.Count()}，耗时：{sw.ElapsedMilliseconds}ms");

            //Assert.IsTrue(markPoints.Any());
        }

        #region Helpers

        //public FaceMarkPoint[] GetFaceMarkPoint(FaceDetector faceDetector, FaceLandmarker faceMark, object bitmap)
        //{
        //    var infos = faceDetector.Detect(bitmap);
        //    var info = infos.First();
        //    return faceMark.Mark(bitmap, info);
        //}

        //public float[] GetExtract(FaceRecognizer faceRecognizer, FaceDetector faceDetector, FaceLandmarker faceMark, object bitmap)
        //{
        //    return faceRecognizer.Extract(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap));
        //}

        public Image ConvertImage(string path)
        {
            return Image.Load(imagePath);
        }
        #endregion
    }
}