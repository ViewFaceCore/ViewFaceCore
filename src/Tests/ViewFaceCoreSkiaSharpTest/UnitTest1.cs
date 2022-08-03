using Microsoft.VisualStudio.TestTools.UnitTesting;
using SkiaSharp;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using ViewFaceCore;
using ViewFaceCore.Configs;
using ViewFaceCore.Core;
using ViewFaceCore.Model;

namespace ViewFaceCoreSkiaSharpTest
{
    [TestClass]
    public class UnitTest1
    {
        private readonly static string imagePath = @"images/Jay_3.jpg";
        private readonly static string imagePath1 = @"images/Jay_4.jpg";
        private readonly static string maskImagePath = @"images/mask_01.jpeg";

        [TestMethod]
        public void DisposableTest()
        {
            FaceDetector faceDetector = new FaceDetector();
            faceDetector.Dispose();

            FaceLandmarker faceMark = new FaceLandmarker();
            faceMark.Dispose();

            FaceRecognizer faceRecognizer = new FaceRecognizer();
            faceRecognizer.Dispose();

            AgePredictor agePredictor = new AgePredictor();
            agePredictor.Dispose();

            EyeStateDetector eyeStateDetector = new EyeStateDetector();
            eyeStateDetector.Dispose();

            GenderPredictor genderPredictor = new GenderPredictor();
            genderPredictor.Dispose();

            FaceAntiSpoofing faceAntiSpoofing = new FaceAntiSpoofing();
            faceAntiSpoofing.Dispose();

            FaceQuality faceQuality = new FaceQuality();
            faceQuality.Dispose();

            FaceTracker faceTrack = new FaceTracker(new FaceTrackerConfig(1920, 1080));
            faceTrack.Dispose();

            MaskDetector maskDetector = new MaskDetector();
            maskDetector.Dispose();

            Assert.IsTrue(true);
        }

        [TestMethod]
        public void ToStringTest()
        {
            using FaceDetector faceDetector = new FaceDetector();

            string name = faceDetector.ToString();

            Assert.IsNotNull(name);
        }


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
            Debug.WriteLine($"{nameof(FaceLandmarker.Mark)}识别，结果：{markPoints.Count()}，耗时：{sw.ElapsedMilliseconds}ms");

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
            Debug.WriteLine($"{QualityType.Brightness}评估，结果：{brightnessResult}，耗时：{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var resolutionResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Resolution);
            Debug.WriteLine($"{QualityType.Resolution}评估，结果：{resolutionResult}，耗时：{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var clarityResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Clarity);
            Debug.WriteLine($"{QualityType.Clarity}评估，结果：{clarityResult}，耗时：{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var clarityExResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.ClarityEx);
            Debug.WriteLine($"{QualityType.ClarityEx}评估，结果：{clarityExResult}，耗时：{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var integrityExResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Integrity);
            Debug.WriteLine($"{QualityType.Integrity}评估，结果：{integrityExResult}，耗时：{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var structureeResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Structure);
            Debug.WriteLine($"{QualityType.Structure}评估，结果：{structureeResult}，耗时：{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var poseResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Pose);
            Debug.WriteLine($"{QualityType.Pose}评估，结果：{poseResult}，耗时：{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var poseExeResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.PoseEx);
            Debug.WriteLine($"{QualityType.PoseEx}评估，结果：{poseExeResult}，耗时：{sw.ElapsedMilliseconds}ms");

            sw.Stop();
            Assert.IsTrue(true);
        }

        /// <summary>
        /// 活体检测测试
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

            var result = faceAntiSpoofing.AntiSpoofing(bitmap, info, markPoints);

            sw.Stop();
            Debug.WriteLine($"{nameof(FaceAntiSpoofing.AntiSpoofing)}检测，结果：{result.Status}，清晰度:{result.Clarity}，真实度：{result.Reality}，耗时：{sw.ElapsedMilliseconds}ms");
        }

        /// <summary>
        /// 人脸追踪测试
        /// </summary>
        [TestMethod]
        public void FaceTrackTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceTracker faceTrack = new FaceTracker(new ViewFaceCore.Configs.FaceTrackerConfig(bitmap.Width, bitmap.Height));

            Stopwatch sw = Stopwatch.StartNew();

            var result = faceTrack.Track(bitmap).ToList();
            sw.Stop();
            Debug.WriteLine($"{nameof(FaceTracker.Track)}追踪，结果：{result.Count()}，耗时：{sw.ElapsedMilliseconds}ms");
            Assert.IsTrue(result.Any());
        }

        /// <summary>
        /// 人脸特征值测试
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
            Debug.WriteLine($"{nameof(FaceRecognizer.Extract)}检测，结果：{result.Count()}，耗时：{sw.ElapsedMilliseconds}ms");
            Assert.IsTrue(result.Any());
        }

        /// <summary>
        /// 年龄预测
        /// </summary>
        [TestMethod]
        public void FaceAgePredictorTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            using AgePredictor agePredictor = new AgePredictor();
            Stopwatch sw = Stopwatch.StartNew();

            var result = agePredictor.PredictAge(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap));
            sw.Stop();
            Debug.WriteLine($"{nameof(AgePredictor.PredictAge)}检测，结果：{result}，耗时：{sw.ElapsedMilliseconds}ms");
            Assert.IsTrue(result > 10);
        }

        /// <summary>
        /// 性别预测
        /// </summary>
        [TestMethod]
        public void FaceGenderPredictorTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            using GenderPredictor genderPredictor = new GenderPredictor();
            Stopwatch sw = Stopwatch.StartNew();

            var result = genderPredictor.PredictGender(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap));

            sw.Stop();
            Debug.WriteLine($"{nameof(GenderPredictor.PredictGender)}检测，结果：{result}，耗时：{sw.ElapsedMilliseconds}ms");
            Assert.IsTrue(result == Gender.Male);
        }

        /// <summary>
        /// 眼睛状态检测
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
            Debug.WriteLine($"{nameof(EyeStateDetector.Detect)}检测，结果：{result.ToString()}，耗时：{sw.ElapsedMilliseconds}ms");
            Assert.IsTrue(result.LeftEyeState == EyeState.Open);
        }

        /// <summary>
        /// 人脸对比测试
        /// </summary>
        [TestMethod]
        public void CompareTest()
        {
            using var bitmap0 = ConvertImage(imagePath);
            using var bitmap1 = ConvertImage(imagePath1);

            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            using FaceRecognizer recognizer = new FaceRecognizer();

            Stopwatch sw = Stopwatch.StartNew();

            var p0 = GetExtract(recognizer, faceDetector, faceMark, bitmap0);
            var p1 = GetExtract(recognizer, faceDetector, faceMark, bitmap1);

            float result = recognizer.Compare(p0, p1);
            bool isSelf = recognizer.IsSelf(p0, p1);
            sw.Stop();
            Debug.WriteLine($"{nameof(FaceRecognizer.Compare)}相似度检测，结果：{result}，是否为同一人：{isSelf}，耗时：{sw.ElapsedMilliseconds}ms");
            Assert.IsTrue(isSelf);
        }

        /// <summary>
        /// 口罩识别测试
        /// </summary>
        [TestMethod]
        public void MaskDetectorTest()
        {
            //using var bitmap_nomask = SKBitmap.Decode(imagePath);
            using var bitmap_mask = SKBitmap.Decode(maskImagePath);

            using MaskDetector maskDetector = new MaskDetector();
            using FaceDetector faceDetector = new FaceDetector();
            //FaceType需要用口罩模型
            using FaceRecognizer faceRecognizer = new FaceRecognizer(new FaceRecognizeConfig()
            {
                FaceType = FaceType.Mask
            });
            using FaceLandmarker faceMark = new FaceLandmarker();


            var info = faceDetector.Detect(bitmap_mask).First();
            bool result = maskDetector.PlotMask(bitmap_mask, info, out float score);

            var p0 = GetExtract(faceRecognizer, faceDetector, faceMark, bitmap_mask);

            Assert.IsTrue(result);
        }

        #region Helpers

        public FaceMarkPoint[] GetFaceMarkPoint(FaceDetector faceDetector, FaceLandmarker faceMark, object bitmap)
        {
            var infos = faceDetector.Detect(bitmap);
            var info = infos.First();
            return faceMark.Mark(bitmap, info);
        }

        public float[] GetExtract(FaceRecognizer faceRecognizer, FaceDetector faceDetector, FaceLandmarker faceMark, object bitmap)
        {
            return faceRecognizer.Extract(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap));
        }

        public SKBitmap ConvertImage(string path)
        {
            return SKBitmap.Decode(imagePath);
        }
        #endregion
    }
}