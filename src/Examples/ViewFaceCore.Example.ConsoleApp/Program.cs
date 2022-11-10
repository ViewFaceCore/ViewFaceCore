using SkiaSharp;
using System;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using ViewFaceCore.Configs;
using ViewFaceCore.Configs.Enums;
using ViewFaceCore.Core;
using ViewFaceCore.Extensions;
using ViewFaceCore.Models;

namespace ViewFaceCore.Example.ConsoleApp
{
    internal class Program
    {
        private readonly static string imagePath0 = @"images/Jay_3.jpg";
        private readonly static string imagePath1 = @"images/Jay_4.jpg";
        private readonly static string maskImagePath = @"images/mask_01.jpeg";

        static void Main(string[] args)
        {
            Console.WriteLine("Hello, ViewFaceCore!\n");

            //人脸识别Demo
            FaceDetectorDemo();

            //关键点标记
            FaceMarkDemo();

            //戴口罩识别Demo
            MaskDetectorDemo();

            //质量检测Demo
            FaceQualityDemo();

            //活体检测Demo
            AntiSpoofingDemo();

            //提取并对比特征值
            FaceRecognizerDemo();

            //人脸追踪
            FaceTrackDemo();

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
            using FaceRecognizer faceRecognizer = new FaceRecognizer(new FaceRecognizeConfig(FaceType.Mask));
            using FaceLandmarker faceMark = new FaceLandmarker();

            var info0 = faceDetector.Detect(bitmap0).First();
            var result0 = maskDetector.Detect(bitmap0, info0);
            Console.WriteLine($"是否戴口罩：{(result0.Status ? "是" : "否")}，置信度：{result0.Score}");

            var info1 = faceDetector.Detect(bitmap_mask).First();
            var result1 = maskDetector.Detect(bitmap_mask, info1);
            Console.WriteLine($"是否戴口罩：{(result1.Status ? "是" : "否")}，置信度：{result1.Score}");

            var result = faceRecognizer.Extract(bitmap_mask, faceMark.Mark(bitmap_mask, info1));
            Console.WriteLine($"是否识别到人脸：{(result != null && result.Sum() > 1 ? "是" : "否")}");
            Console.WriteLine();
        }

        static void FaceMarkDemo()
        {
            using var bitmap0 = SKBitmap.Decode(imagePath0);
            using var faceImage = bitmap0.ToFaceImage();
            using FaceDetector faceDetector = new FaceDetector();
            using FaceLandmarker faceMark = new FaceLandmarker();
            Stopwatch sw = new Stopwatch();

            var infos = faceDetector.Detect(faceImage);
            var markPoints = faceMark.Mark(faceImage, infos[0]);

            sw.Stop();
            Console.WriteLine($"识别到的关键点个数：{markPoints.Length}，耗时：{sw.ElapsedMilliseconds}ms");
            foreach (var item in markPoints)
            {
                Console.WriteLine($"X:{item.X}\tY:{item.Y}");
            }
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

            var result = faceAntiSpoofing.Predict(bitmap, info, markPoints);
            Console.WriteLine($"活体检测，结果：{result.Status}，清晰度:{result.Clarity}，真实度：{result.Reality}，耗时：{sw.ElapsedMilliseconds}ms");

            sw.Stop();
            Console.WriteLine();
        }

        static void FaceRecognizerDemo()
        {
            Stopwatch sw = Stopwatch.StartNew();

            using var faceImage0 = SKBitmap.Decode(imagePath0).ToFaceImage();
            using var faceImage1 = SKBitmap.Decode(imagePath1).ToFaceImage();
            //检测人脸信息
            using FaceDetector faceDetector = new FaceDetector();
            FaceInfo[] infos0 = faceDetector.Detect(faceImage0);
            FaceInfo[] infos1 = faceDetector.Detect(faceImage1);
            //标记人脸位置
            using FaceLandmarker faceMark = new FaceLandmarker();
            FaceMarkPoint[] points0 = faceMark.Mark(faceImage0, infos0[0]);
            FaceMarkPoint[] points1 = faceMark.Mark(faceImage1, infos1[0]);
            //提取特征值
            using FaceRecognizer faceRecognizer = new FaceRecognizer();
            float[] data0 = faceRecognizer.Extract(faceImage0, points0);
            float[] data1 = faceRecognizer.Extract(faceImage1, points1);
            //对比特征值
            bool isSelf = faceRecognizer.IsSelf(data0, data1);

            Console.WriteLine($"识别到的人脸是否为同一人：{isSelf}，对比耗时：{sw.ElapsedMilliseconds}ms");
            Console.WriteLine();
            sw.Stop();
        }

        static void FaceTrackDemo()
        {
            using var faceImage = SKBitmap.Decode(imagePath0).ToFaceImage();
            using FaceLandmarker faceMark = new FaceLandmarker();
            using FaceTracker faceTrack = new FaceTracker(new FaceTrackerConfig(faceImage.Width, faceImage.Height));
            var infos = faceTrack.Track(faceImage);
            if (infos == null || !infos.Any())
            {
                Console.WriteLine("未追踪到任何人脸！");
                return;
            }
            Console.WriteLine($"追踪到的人脸数量：{infos.Length} 个人脸信息：\n");
            Console.WriteLine($"No.\t人脸置信度\t位置信息");
            for (int i = 0; i < infos.Length; i++)
            {
                Console.WriteLine($"{i}\t{infos[i].Score:f8}\t{infos[i].Location}");
            }
            Console.WriteLine();
        }
    }
}