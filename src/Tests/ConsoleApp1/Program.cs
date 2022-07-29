using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using ViewFaceCore;
using ViewFaceCore.Model;

namespace ConsoleApp1
{
    internal class Program
    {
        private readonly static NLog.Logger logger = NLog.LogManager.GetCurrentClassLogger();
        private readonly static string imagePath = @"images\Jay_3.jpg";

        static void Main(string[] args)
        {
            //人脸识别和标记测试，开始：2022-07-28 17:11，结束：,结果：
            //FaceDetectorAndFaceMarkTest();

            //活体检测测试，通过24h测试，20220728
            //AntiSpoofingTest();

            //质量评估测试，开始：2022-07-28 09:57，结束：,结果：不通过，有三个评估方式有问题
            FaceQualityTest();

            Console.WriteLine("Hello, World!");
        }

        private static IEnumerable<FaceMarkPoint> GetFaceMarkPoint(ViewFace viewFace, SKBitmap bitmap)
        {
            var infos = viewFace.FaceDetector(bitmap);
            var info = infos.First();
            return viewFace.FaceMark(bitmap, info).ToList();
        }

        private static void FaceDetectorAndFaceMarkTest()
        {
            using SKBitmap bitmap = SKBitmap.Decode(imagePath);
            ViewFace viewFace = new ViewFace();
            Stopwatch sw = new Stopwatch();
            sw.Start();
            int i = 0;
            while (true)
            {
                sw.Restart();
                var infos = viewFace.FaceDetector(bitmap);
                var info = infos.First();
                var markPoints = GetFaceMarkPoint(viewFace, bitmap);

                logger.Info($"第{i + 1}次识别，结果：{markPoints.Count()}，耗时：{sw.ElapsedMilliseconds}ms");
                sw.Stop();
                i++;
            }
        }

        private static void FaceQualityTest()
        {
            using SKBitmap bitmap = SKBitmap.Decode(imagePath);
            ViewFace viewFace = new ViewFace();
            var infos = viewFace.FaceDetector(bitmap);
            var info = infos.First();
            var markPoints = GetFaceMarkPoint(viewFace, bitmap);

            Stopwatch sw = new Stopwatch();
            sw.Start();
            int i = 0;
            while (true)
            {
                sw.Restart();
                //var brightnessResult = viewFace.FaceQuality(bitmap, info, markPoints, QualityType.Brightness);
                //logger.Info($"第{i + 1}次{QualityType.Brightness}评估，结果：{brightnessResult}，耗时：{sw.ElapsedMilliseconds}ms");
                //var resolutionResult = viewFace.FaceQuality(bitmap, info, markPoints, QualityType.Resolution);
                //logger.Info($"第{i + 1}次{QualityType.Resolution}评估，结果：{resolutionResult}，耗时：{sw.ElapsedMilliseconds}ms");
                //var clarityResult = viewFace.FaceQuality(bitmap, info, markPoints, QualityType.Clarity);
                //logger.Info($"第{i + 1}次{QualityType.Clarity}评估，结果：{clarityResult}，耗时：{sw.ElapsedMilliseconds}ms");
                //var clarityExResult = viewFace.FaceQuality(bitmap, info, markPoints, QualityType.ClarityEx);
                //logger.Info($"第{i + 1}次{QualityType.ClarityEx}评估，结果：{clarityExResult}，耗时：{sw.ElapsedMilliseconds}ms");
                //var integrityExResult = viewFace.FaceQuality(bitmap, info, markPoints, QualityType.Integrity);
                //logger.Info($"第{i + 1}次{QualityType.Integrity}评估，结果：{integrityExResult}，耗时：{sw.ElapsedMilliseconds}ms");

                //
                var poseResult = viewFace.FaceQuality(bitmap, info, markPoints, QualityType.Pose);
                logger.Info($"第{i + 1}次{QualityType.Pose}评估，结果：{poseResult}，耗时：{sw.ElapsedMilliseconds}ms");
                //var poseExeResult = viewFace.FaceQuality(bitmap, info, markPoints, QualityType.PoseEx);
                //logger.Info($"第{i + 1}次{QualityType.PoseEx}评估，结果：{poseExeResult}，耗时：{sw.ElapsedMilliseconds}ms");
                //var structureeResult = viewFace.FaceQuality(bitmap, info, markPoints, QualityType.Structure);
                //logger.Info($"第{i + 1}次{QualityType.Structure}评估，结果：{structureeResult}，耗时：{sw.ElapsedMilliseconds}ms");
                sw.Stop();
                i++;
            }
        }

        private static void AntiSpoofingTest()
        {
            using SKBitmap bitmap = SKBitmap.Decode(imagePath);
            ViewFace viewFace = new ViewFace();
            var infos = viewFace.FaceDetector(bitmap);
            var info = infos.First();
            var markPoints = GetFaceMarkPoint(viewFace, bitmap);

            Stopwatch sw = new Stopwatch();
            sw.Start();
            int i = 0;
            while (true)
            {
                sw.Restart();
                var result = viewFace.AntiSpoofing(bitmap, info, markPoints);
                logger.Info($"第{i + 1}次检测，结果：{result}，耗时：{sw.ElapsedMilliseconds}ms");
                sw.Stop();
                i++;
            }
        }

        private static void ExtractTest()
        {

        }
    }
}