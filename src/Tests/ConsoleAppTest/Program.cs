using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using ViewFaceCore;
using ViewFaceCore.Configs;
using ViewFaceCore.Core;
using ViewFaceCore.Models;
using ViewFaceCore.Extensions;
using ViewFaceCore.Configs.Enums;
using System.Drawing;

namespace ConsoleAppTest
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
                //MaskDetectorTest();

                //////人脸识别和标记测试，开始：2022/07/30 00:12:51，结束：2022/07/30 09:04:30，结果：通过
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
                FaceAgePredictorTest();

                ////性别预测测试
                FaceGenderPredictorTest();

                ////眼睛状态检测测试
                //FaceEyeStateDetectorTest();

                ////人脸对比测试，开始：2022 / 07 / 30 00:12:51，结束：2022 / 07 / 30 09:04:30，结果：通过
                //CompareTest();
            }
        }

        private static void FaceDetectorAndFaceMarkTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();

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
                var result = faceAntiSpoofing.Predict(bitmap, info, markPoints);
                logger.Info($"第{i + 1}次{nameof(FaceAntiSpoofing.Predict)}检测，结果：{result.Status}，清晰度:{result.Clarity}，真实度：{result.Reality}，耗时：{sw.ElapsedMilliseconds}ms");
            });
        }

        /// <summary>
        /// 人脸追踪测试
        /// </summary>
        private static void FaceTrackTest()
        {
            using var bitmap = ConvertImage(imagePath);
            using FaceLandmarker faceMark = new FaceLandmarker();
            using FaceTracker faceTrack = new FaceTracker(new ViewFaceCore.Configs.FaceTrackerConfig(bitmap.Width, bitmap.Height));
            Worker((sw, i) =>
            {
                var result = faceTrack.Track(bitmap).ToList();
                if (result == null || !result.Any())
                {
                    Console.WriteLine("GG...");
                    return;
                }
                foreach (var item in result)
                {
                    var points = faceMark.Mark(bitmap, item);
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
            using FaceDetector faceDetector = new FaceDetector(new FaceDetectConfig()
            {
                DeviceType = DeviceType.AUTO
            });
            using FaceLandmarker faceMark = new FaceLandmarker(new FaceLandmarkConfig()
            {
                DeviceType = DeviceType.AUTO
            });
            using FaceRecognizer faceRecognizer = new FaceRecognizer(new FaceRecognizeConfig()
            {
                DeviceType = DeviceType.AUTO
            });
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
                var result = agePredictor.PredictAge(bitmap);
                logger.Info($"第{i + 1}次{nameof(AgePredictor.PredictAge)}检测，结果：{result}，耗时：{sw.ElapsedMilliseconds}ms");
                sw.Restart();
                result = agePredictor.PredictAgeWithCrop(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap));
                logger.Info($"第{i + 1}次{nameof(AgePredictor.PredictAgeWithCrop)}检测，结果：{result}，耗时：{sw.ElapsedMilliseconds}ms");
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
                var faceInfo = faceDetector.Detect(bitmap)[0];
                using Bitmap bitmap1 = CutPicture(bitmap, faceInfo.Location.X, faceInfo.Location.Y, faceInfo.Location.Width, faceInfo.Location.Height);
                var result = genderPredictor.PredictGender(bitmap1);
                logger.Info($"第{i + 1}次{nameof(GenderPredictor.PredictGender)}检测，结果：{result}，耗时：{sw.ElapsedMilliseconds}ms");
                sw.Restart();
                result = genderPredictor.PredictGenderWithCrop(bitmap, GetFaceMarkPoint(faceDetector, faceMark, bitmap));
                logger.Info($"第{i + 1}次{nameof(GenderPredictor.PredictGenderWithCrop)}检测，结果：{result}，耗时：{sw.ElapsedMilliseconds}ms");
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

            using MaskDetector maskDetector = new MaskDetector();
            using FaceDetector faceDetector = new FaceDetector();
            //FaceType需要用口罩模型
            using FaceRecognizer faceRecognizer = new FaceRecognizer(new FaceRecognizeConfig()
            {
                FaceType = FaceType.Mask
            });

            var info = faceDetector.Detect(bitmap_mask).First();

            Worker((sw, i) =>
            {
                var result = maskDetector.Detect(bitmap_mask, info);
                logger.Info($"第{i + 1}次{nameof(MaskDetector.Detect)}戴口罩检测，结果：{result.Status}，置信度：{result.Score}，耗时：{sw.ElapsedMilliseconds}ms");
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

        public static Bitmap ConvertImage(string path)
        {
            return (Bitmap)Bitmap.FromFile(imagePath);
        }

        /// 图片裁剪，生成新图，保存在同一目录下,名字加_new，格式1.png  新图1_new.png
        /// </summary>
        /// <param name="picPath">要修改图片完整路径</param>
        /// <param name="x">修改起点x坐标</param>
        /// <param name="y">修改起点y坐标</param>
        /// <param name="width">新图宽度</param>
        /// <param name="height">新图高度</param>
        public static Bitmap CutPicture(Bitmap img, int x, int y, int width, int height)
        {
            //定义截取矩形
            System.Drawing.Rectangle cropArea = new System.Drawing.Rectangle(x, y, width, height);
            //要截取的区域大小
            //判断超出的位置否
            if ((img.Width < x + width) || img.Height < y + height)
            {
                img.Dispose();
                return null;
            }
            //定义Bitmap对象
            System.Drawing.Bitmap bmpImage = new System.Drawing.Bitmap(img);
            //进行裁剪
            System.Drawing.Bitmap bmpCrop = bmpImage.Clone(cropArea, bmpImage.PixelFormat);
            //释放对象
            img.Dispose();
            return bmpCrop;
        }

        #endregion
    }
}