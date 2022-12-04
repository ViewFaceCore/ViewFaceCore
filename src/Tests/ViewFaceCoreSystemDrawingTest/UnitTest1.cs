using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Threading;
using ViewFaceCore;
using ViewFaceCore.Core;
using ViewFaceCore.Models;

namespace ViewFaceCoreSystemDrawingTest
{
    [TestClass]
    public class UnitTest1
    {
        private readonly static string imagePath = @"images/Jay_3.jpg";
        private readonly static string imagePath1 = @"images/Jay_4.jpg";

        [TestMethod]
        public void FaceDetectorAndFaceMarkTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();

            Stopwatch sw = Stopwatch.StartNew();

            var infos = faceDetector.Detect(bitmap);
            var info = infos.First();
            var markPoints = GetFaceMarkPoint(faceDetector, faceMark, bitmap);

            sw.Stop();
            Debug.WriteLine($"{nameof(FaceLandmarker.Mark)}ʶ�𣬽����{markPoints.Count()}����ʱ��{sw.ElapsedMilliseconds}ms");

            Assert.IsTrue(markPoints.Any());
        }

        [TestMethod]
        public void FaceQualityTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceQuality faceQuality = new FaceQuality();
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();

            var info = faceDetector.Detect(bitmap).First();
            var markPoints = GetFaceMarkPoint(faceDetector, faceMark, bitmap);

            Stopwatch sw = Stopwatch.StartNew();

            var brightnessResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Brightness);
            Debug.WriteLine($"{QualityType.Brightness}�����������{brightnessResult}����ʱ��{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var resolutionResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Resolution);
            Debug.WriteLine($"{QualityType.Resolution}�����������{resolutionResult}����ʱ��{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var clarityResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Clarity);
            Debug.WriteLine($"{QualityType.Clarity}�����������{clarityResult}����ʱ��{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var clarityExResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.ClarityEx);
            Debug.WriteLine($"{QualityType.ClarityEx}�����������{clarityExResult}����ʱ��{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var integrityExResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Integrity);
            Debug.WriteLine($"{QualityType.Integrity}�����������{integrityExResult}����ʱ��{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var structureeResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Structure);
            Debug.WriteLine($"{QualityType.Structure}�����������{structureeResult}����ʱ��{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var poseResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Pose);
            Debug.WriteLine($"{QualityType.Pose}�����������{poseResult}����ʱ��{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var poseExeResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.PoseEx);
            Debug.WriteLine($"{QualityType.PoseEx}�����������{poseExeResult}����ʱ��{sw.ElapsedMilliseconds}ms");

            sw.Stop();
            Assert.IsTrue(true);
        }

        /// <summary>
        /// ���������
        /// </summary>
        [TestMethod]
        public void AntiSpoofingTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            using FaceAntiSpoofing faceAntiSpoofing = new FaceAntiSpoofing();
            var info = faceDetector.Detect(bitmap).First();
            var markPoints = GetFaceMarkPoint(faceDetector, faceMark, bitmap);

            Stopwatch sw = Stopwatch.StartNew();

            var result = faceAntiSpoofing.Predict(bitmap, info, markPoints);

            sw.Stop();
            Debug.WriteLine($"{nameof(FaceAntiSpoofing.Predict)}��⣬�����{result.Status}��������:{result.Clarity}����ʵ�ȣ�{result.Reality}����ʱ��{sw.ElapsedMilliseconds}ms");
        }

        /// <summary>
        /// ����׷�ٲ���
        /// </summary>
        [TestMethod]
        public void FaceTrackTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceTracker faceTrack = new FaceTracker(new ViewFaceCore.Configs.FaceTrackerConfig(bitmap.Width, bitmap.Height));

            Stopwatch sw = Stopwatch.StartNew();

            var result = faceTrack.Track(bitmap).ToList();
            sw.Stop();
            Debug.WriteLine($"{nameof(FaceTracker.Track)}׷�٣������{result.Count()}����ʱ��{sw.ElapsedMilliseconds}ms");
            Assert.IsTrue(result.Any());
        }

        /// <summary>
        /// ��������ֵ����
        /// </summary>
        [TestMethod]
        public void ExtractTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            using FaceRecognizer faceRecognizer = new FaceRecognizer();

            Stopwatch sw = Stopwatch.StartNew();

            var result = faceRecognizer.Extract(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap)).ToList();

            sw.Stop();
            Debug.WriteLine($"{nameof(FaceRecognizer.Extract)}��⣬�����{result.Count()}����ʱ��{sw.ElapsedMilliseconds}ms");
            Assert.IsTrue(result.Any());
        }

        /// <summary>
        /// ����Ԥ��
        /// </summary>
        [TestMethod]
        public void FaceAgePredictorTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            using AgePredictor agePredictor = new AgePredictor();
            Stopwatch sw = Stopwatch.StartNew();

            var result = agePredictor.PredictAgeWithCrop(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap));
            sw.Stop();
            Debug.WriteLine($"{nameof(AgePredictor.PredictAge)}��⣬�����{result}����ʱ��{sw.ElapsedMilliseconds}ms");
            Assert.IsTrue(result > 10);
        }

        /// <summary>
        /// �Ա�Ԥ��
        /// </summary>
        [TestMethod]
        public void FaceGenderPredictorTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            using GenderPredictor genderPredictor = new GenderPredictor();
            Stopwatch sw = Stopwatch.StartNew();

            var result = genderPredictor.PredictGenderWithCrop(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap));

            sw.Stop();
            Debug.WriteLine($"{nameof(GenderPredictor.PredictGender)}��⣬�����{result}����ʱ��{sw.ElapsedMilliseconds}ms");
            Assert.IsTrue(result == Gender.Male);
        }

        /// <summary>
        /// �۾�״̬���
        /// </summary>
        [TestMethod]
        public void FaceEyeStateDetectorTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            using EyeStateDetector eyeStateDetector = new EyeStateDetector();
            Stopwatch sw = Stopwatch.StartNew();

            var result = eyeStateDetector.Detect(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap));
            sw.Stop();
            Debug.WriteLine($"{nameof(EyeStateDetector.Detect)}��⣬�����{result.ToString()}����ʱ��{sw.ElapsedMilliseconds}ms");
            Assert.IsTrue(result.LeftEyeState == EyeState.Open);
        }

        /// <summary>
        /// �����ԱȲ���
        /// </summary>
        [TestMethod]
        public void CompareTest()
        {
            using var bitmap0 = ConvertImage(imagePath);
            using var bitmap1 = ConvertImage(imagePath1);

            using (FaceDetector faceDetector = new FaceDetector())
            {
                using (FaceLandmarker faceMark = new FaceLandmarker())
                {
                    using (FaceRecognizer recognizer = new FaceRecognizer())
                    {
                        Stopwatch sw = Stopwatch.StartNew();

                        var p0 = GetExtract(recognizer, faceDetector, faceMark, bitmap0);
                        var p1 = GetExtract(recognizer, faceDetector, faceMark, bitmap1);

                        float result = recognizer.Compare(p0, p1);
                        bool isSelf = recognizer.IsSelf(p0, p1);
                        sw.Stop();
                        Debug.WriteLine($"{nameof(FaceRecognizer.Compare)}���ƶȼ�⣬�����{result}���Ƿ�Ϊͬһ�ˣ�{isSelf}����ʱ��{sw.ElapsedMilliseconds}ms");
                        Assert.IsTrue(isSelf);
                    }
                }
            }
            while (true)
            {
                Thread.Sleep(1000);
            }
        }

        #region Helpers

        public FaceMarkPoint[] GetFaceMarkPoint(FaceDetector faceDetector, FaceLandmarker faceMark, Bitmap bitmap)
        {
            var infos = faceDetector.Detect(bitmap);
            var info = infos.First();
            return faceMark.Mark(bitmap, info);
        }

        public float[] GetExtract(FaceRecognizer faceRecognizer, FaceDetector faceDetector, FaceLandmarker faceMark, Bitmap bitmap)
        {
            return faceRecognizer.Extract(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap));
        }

        public Bitmap ConvertImage(string path)
        {
            return (Bitmap)Image.FromFile(imagePath);
        }
        #endregion
    }
}