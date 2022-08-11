using SixLabors.ImageSharp;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using ViewFaceCore;
using ViewFaceCore.Configs;
using ViewFaceCore.Core;
using ViewFaceCore.Model;

namespace ConsoleApp1
{
    internal class Program
    {
        private readonly static NLog.Logger logger = NLog.LogManager.GetCurrentClassLogger();
        private readonly static string imagePath = @"images/Jay_3.jpg";
        private readonly static string imagePath1 = @"images/Jay_4.jpg";
        private readonly static string maskImagePath = @"images/mask_01.jpeg";
        private readonly static string logPath = "logs";

        static void Main(string[] args)
        {
            if (Directory.Exists(logPath))
            {
                Directory.Delete(logPath, true);
            }
            while (true)
            {
                //口罩识别测试
                MaskDetectorTest();

                ////人脸识别和标记测试，开始：2022/07/30 00:12:51，结束：2022/07/30 09:04:30，结果：通过
                //FaceDetectorAndFaceMarkTest();

                ////活体检测测试，通过24h测试，20220728
                //AntiSpoofingTest();

                ////质量评估测试，开始：2022 - 07 - 28 09:57，结束：,结果：通过
                //FaceQualityTest();

                ////人脸追踪测试，开始：2022/07/29 16:45:18，结束：2022/07/29 17:50:01,结果：通过
                //FaceTrackTest();

                ////人脸特征值测试，开始：2022/07/30 00:12:51，结束：2022/07/30 09:04:30，结果：通过
                //ExtractTest();

                ////年龄预测测试
                //FaceAgePredictorTest();

                ////性别预测测试
                //FaceGenderPredictorTest();

                ////眼睛状态检测测试
                //FaceEyeStateDetectorTest();

                //人脸对比测试，开始：2022/07/30 00:12:51，结束：2022/07/30 09:04:30，结果：通过
                //CompareTest();
            }

            Console.WriteLine("Hello, World!");
        }

        private static void FaceDetectorAndFaceMarkTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            Stopwatch sw = new Stopwatch();

            Worker((sw, i) =>
            {
                var infos = faceDetector.Detect(bitmap);
                var info = infos.First();
                var markPoints = GetFaceMarkPoint(faceDetector, faceMark, bitmap);

                logger.Info($"第{i + 1}次{nameof(FaceLandmarker.Mark)}识别，结果：{markPoints.Length}，耗时：{sw.ElapsedMilliseconds}ms");
            });
        }

        private static void FaceQualityTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceQuality faceQuality = new FaceQuality();
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();

            var info = faceDetector.Detect(bitmap).First();
            var markPoints = GetFaceMarkPoint(faceDetector, faceMark, bitmap);

            Worker((sw, i) =>
            {
                var brightnessResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Brightness);
                logger.Info($"第{i + 1}次{QualityType.Brightness}评估，结果：{brightnessResult}，耗时：{sw.ElapsedMilliseconds}ms");
                sw.Restart();
                var resolutionResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Resolution);
                logger.Info($"第{i + 1}次{QualityType.Resolution}评估，结果：{resolutionResult}，耗时：{sw.ElapsedMilliseconds}ms");
                sw.Restart();
                var clarityResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Clarity);
                logger.Info($"第{i + 1}次{QualityType.Clarity}评估，结果：{clarityResult}，耗时：{sw.ElapsedMilliseconds}ms");
                sw.Restart();
                var clarityExResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.ClarityEx);
                logger.Info($"第{i + 1}次{QualityType.ClarityEx}评估，结果：{clarityExResult}，耗时：{sw.ElapsedMilliseconds}ms");
                sw.Restart();
                var integrityExResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Integrity);
                logger.Info($"第{i + 1}次{QualityType.Integrity}评估，结果：{integrityExResult}，耗时：{sw.ElapsedMilliseconds}ms");
                sw.Restart();
                var structureeResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Structure);
                logger.Info($"第{i + 1}次{QualityType.Structure}评估，结果：{structureeResult}，耗时：{sw.ElapsedMilliseconds}ms");
                sw.Restart();
                var poseResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Pose);
                logger.Info($"第{i + 1}次{QualityType.Pose}评估，结果：{poseResult}，耗时：{sw.ElapsedMilliseconds}ms");
                sw.Restart();
                var poseExeResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.PoseEx);
                logger.Info($"第{i + 1}次{QualityType.PoseEx}评估，结果：{poseExeResult}，耗时：{sw.ElapsedMilliseconds}ms");

            });
        }

        /// <summary>
        /// 活体检测测试
        /// </summary>
        private static void AntiSpoofingTest()
        {
            using var bitmap = ConvertImage(maskImagePath);
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            logger.Info("开始加载活体识别....");

            var info = faceDetector.Detect(bitmap).First();
            var markPoints = GetFaceMarkPoint(faceDetector, faceMark, bitmap);

            using FaceAntiSpoofing faceAntiSpoofing = new FaceAntiSpoofing(new FaceAntiSpoofingConfig()
            {
                Global = true
            });

            Worker((sw, i) =>
            {
                var result = faceAntiSpoofing.AntiSpoofing(bitmap, info, markPoints);
                logger.Info($"第{i + 1}次{nameof(FaceAntiSpoofing.AntiSpoofing)}检测，结果：{result.Status}，清晰度:{result.Clarity}，真实度：{result.Reality}，耗时：{sw.ElapsedMilliseconds}ms");
            });
        }

        /// <summary>
        /// 人脸追踪测试
        /// </summary>
        private static void FaceTrackTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceTracker faceTrack = new FaceTracker(new ViewFaceCore.Configs.FaceTrackerConfig(bitmap.Width, bitmap.Height));
            Worker((sw, i) =>
            {
                var result = faceTrack.Track(bitmap).ToList();
                if (result == null || !result.Any())
                {
                    Console.WriteLine("GG...");
                    return;
                }
                logger.Info($"第{i + 1}次{nameof(FaceTracker.Track)}追踪，结果：{result.Count()}，耗时：{sw.ElapsedMilliseconds}ms");
            });
        }

        /// <summary>
        /// 人脸特征值测试
        /// </summary>
        private static void ExtractTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            using FaceRecognizer faceRecognizer = new FaceRecognizer();
            Worker((sw, i) =>
            {
                var result = faceRecognizer.Extract(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap)).ToList();
                logger.Info($"第{i + 1}次{nameof(FaceRecognizer.Extract)}检测，结果：{result.Count()}，耗时：{sw.ElapsedMilliseconds}ms");
            });
        }

        /// <summary>
        /// 年龄预测
        /// </summary>
        private static void FaceAgePredictorTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            using AgePredictor agePredictor = new AgePredictor();
            Worker((sw, i) =>
            {
                var result = agePredictor.PredictAge(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap));
                logger.Info($"第{i + 1}次{nameof(AgePredictor.PredictAge)}检测，结果：{result}，耗时：{sw.ElapsedMilliseconds}ms");
            });
        }

        /// <summary>
        /// 性别预测
        /// </summary>
        private static void FaceGenderPredictorTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            using GenderPredictor genderPredictor = new GenderPredictor();
            Worker((sw, i) =>
            {
                var result = genderPredictor.PredictGender(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap));
                logger.Info($"第{i + 1}次{nameof(GenderPredictor.PredictGender)}检测，结果：{result}，耗时：{sw.ElapsedMilliseconds}ms");
            });
        }

        /// <summary>
        /// 眼睛状态检测
        /// </summary>
        private static void FaceEyeStateDetectorTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            using EyeStateDetector eyeStateDetector = new EyeStateDetector();
            Worker((sw, i) =>
            {
                var result = eyeStateDetector.Detect(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap));
                logger.Info($"第{i + 1}次{nameof(EyeStateDetector.Detect)}检测，结果：{result.ToString()}，耗时：{sw.ElapsedMilliseconds}ms");
            });
        }

        /// <summary>
        /// 人脸对比测试
        /// </summary>
        private static void CompareTest()
        {
            using var bitmap0 = ConvertImage(imagePath);
            using var bitmap1 = ConvertImage(imagePath1);

            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            using FaceRecognizer recognizer = new FaceRecognizer();

            Worker((sw, i) =>
            {
                var p0 = GetExtract(recognizer, faceDetector, faceMark, bitmap0);
                var p1 = GetExtract(recognizer, faceDetector, faceMark, bitmap1);

                float result = recognizer.Compare(p0, p1);
                bool isSelf = recognizer.IsSelf(p0, p1);
                logger.Info($"第{i + 1}次{nameof(FaceRecognizer.Compare)}相似度检测，结果：{result}，是否为同一人：{isSelf}，耗时：{sw.ElapsedMilliseconds}ms");
            });

        }

        /// <summary>
        /// 
        /// </summary>
        private static void MaskDetectorTest()
        {
            if (!File.Exists(imagePath))
            {
                throw new Exception("图像不存在！");
            }
            using var bitmap_nomask = ConvertImage(imagePath);
            using var bitmap_mask = ConvertImage(maskImagePath);

            using MaskDetector maskDetector = new MaskDetector(new MaskDetectConfig()
            {
                LogEvent = (string log) =>
                {
                    logger.Info(log);
                }
            });
            using FaceDetector faceDetector = new FaceDetector();
            //FaceType需要用口罩模型
            using FaceRecognizer faceRecognizer = new FaceRecognizer(new FaceRecognizeConfig()
            {
                FaceType = FaceType.Mask
            });

            var info = faceDetector.Detect(bitmap_mask).First();

            Worker((sw, i) =>
            {
                var result = maskDetector.PlotMask(bitmap_mask, info);
                logger.Info($"第{i + 1}次{nameof(MaskDetector.PlotMask)}戴口罩检测，结果：{result.Status}，置信度：{result.Score}，耗时：{sw.ElapsedMilliseconds}ms");
            });

        }

        #region Helpers

        private static FaceMarkPoint[] GetFaceMarkPoint(FaceDetector faceDetector, FaceLandmarker faceMark, Image bitmap)
        {
            var infos = faceDetector.Detect(bitmap);
            var info = infos.First();
            return faceMark.Mark(bitmap, info);
        }

        private static float[] GetExtract(FaceRecognizer faceRecognizer, FaceDetector faceDetector, FaceLandmarker faceMark, Image bitmap)
        {
            return faceRecognizer.Extract(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap));
        }

        private static void Worker(Action<Stopwatch, int> action)
        {
            Stopwatch sw = new Stopwatch();

            Stopwatch sw2 = new Stopwatch();
            sw2.Start();

            sw.Start();
            int i = 0;
            while (true)
            {
                sw.Restart();
                action(sw, i);
                sw.Stop();
                i++;

                if (sw2.ElapsedMilliseconds > 1 * 3 * 1000)
                {
                    break;
                }
            }
            sw2.Stop();
        }

        public static Image ConvertImage(string path)
        {
            return Image.Load(imagePath);
        }

        #endregion
    }
}