using SkiaSharp;
using System;
using System.Diagnostics;
using System.Linq;
using ViewFaceCore.Configs;
using ViewFaceCore.Core;
using ViewFaceCore.Model;

namespace ViewFaceCore.Demo.ConsoleApp
{
    internal class Program
    {
        private readonly static string imagePath0 = @"images/Jay_3.jpg";
        private readonly static string imagePath1 = @"images/Jay_4.jpg";
        private readonly static string maskImagePath = @"images/mask_01.jpeg";

        static void Main(string[] args)
        {
            Console.WriteLine("Hello, ViewFaceCore!");

            //人脸识别Demo
            FaceDetectorDemo();

            //戴口罩识别Demo
            MaskDetectorDemo();

            //质量检测Demo
            FaceQualityDemo();

            //活体检测Demo
            AntiSpoofingDemo();

            Console.ReadKey();
        }

        static void FaceDetectorDemo()
        {
            using var bitmap = SKBitmap.Decode(imagePath0);
            using FaceDetector faceDetector = new FaceDetector();
            FaceInfo[] infos = faceDetector.Detect(bitmap);
            Console.WriteLine($"识别到的人脸数量：{infos.Length} 个人脸信息：\n");
            Console.WriteLine($"No.\t人脸置信度\t位置信息");
            for (int i = 0; i < infos.Length; i++)
            {
                Console.WriteLine($"{i}\t{infos[i].Score:f8}\t{infos[i].Location}");
            }
            Console.WriteLine();
        }

        static void MaskDetectorDemo()
        {
            using var bitmap0 = SKBitmap.Decode(imagePath0);
            using var bitmap_mask = SKBitmap.Decode(maskImagePath);

            using MaskDetector maskDetector = new MaskDetector();
            using FaceDetector faceDetector = new FaceDetector();
            //FaceType需要用口罩模型
            using FaceRecognizer faceRecognizer = new FaceRecognizer(new FaceRecognizeConfig()
            {
                FaceType = FaceType.Mask
            });
            using FaceLandmarker faceMark = new FaceLandmarker();

            var info0 = faceDetector.Detect(bitmap0).First();
            var result0 = maskDetector.PlotMask(bitmap0, info0);
            Console.WriteLine($"是否戴口罩：{(result0.Status ? "是" : "否")}，置信度：{result0.Score}");

            var info1 = faceDetector.Detect(bitmap_mask).First();
            var result1 = maskDetector.PlotMask(bitmap_mask, info1);
            Console.WriteLine($"是否戴口罩：{(result1.Status ? "是" : "否")}，置信度：{result1.Score}");

            var result = faceRecognizer.Extract(bitmap_mask, faceMark.Mark(bitmap_mask, info1));
            Console.WriteLine($"是否识别到人脸：{(result != null && result.Sum() > 1 ? "是" : "否")}");
            Console.WriteLine();
        }

        static void FaceQualityDemo()
        {
            using var bitmap = SKBitmap.Decode(imagePath0);
            using FaceQuality faceQuality = new FaceQuality();
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();

            var info = faceDetector.Detect(bitmap).First();
            var markPoints = faceMark.Mark(bitmap, info);

            Stopwatch sw = Stopwatch.StartNew();

            var brightnessResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Brightness);
            Console.WriteLine($"{QualityType.Brightness}评估，结果：{brightnessResult}，耗时：{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var resolutionResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Resolution);
            Console.WriteLine($"{QualityType.Resolution}评估，结果：{resolutionResult}，耗时：{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var clarityResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Clarity);
            Console.WriteLine($"{QualityType.Clarity}评估，结果：{clarityResult}，耗时：{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var clarityExResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.ClarityEx);
            Console.WriteLine($"{QualityType.ClarityEx}评估，结果：{clarityExResult}，耗时：{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var integrityExResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Integrity);
            Console.WriteLine($"{QualityType.Integrity}评估，结果：{integrityExResult}，耗时：{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var structureeResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Structure);
            Console.WriteLine($"{QualityType.Structure}评估，结果：{structureeResult}，耗时：{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var poseResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.Pose);
            Console.WriteLine($"{QualityType.Pose}评估，结果：{poseResult}，耗时：{sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var poseExeResult = faceQuality.Detect(bitmap, info, markPoints, QualityType.PoseEx);
            Console.WriteLine($"{QualityType.PoseEx}评估，结果：{poseExeResult}，耗时：{sw.ElapsedMilliseconds}ms");

            sw.Stop();
            Console.WriteLine();
        }

        static void AntiSpoofingDemo()
        {
            using var bitmap = SKBitmap.Decode(imagePath0);

            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            using FaceAntiSpoofing faceAntiSpoofing = new FaceAntiSpoofing();

            var info = faceDetector.Detect(bitmap).First();
            var markPoints = faceMark.Mark(bitmap, info);

            Stopwatch sw = Stopwatch.StartNew();
            sw.Start();

            var result = faceAntiSpoofing.AntiSpoofing(bitmap, info, markPoints);
            Console.WriteLine($"活体检测，结果：{result.Status}，清晰度:{result.Clarity}，真实度：{result.Reality}，耗时：{sw.ElapsedMilliseconds}ms");

            sw.Stop();
            Console.WriteLine();
        }
    }
}