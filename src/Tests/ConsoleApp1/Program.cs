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
            Console.WriteLine("Hello, World!");
        }

        private static IEnumerable<FaceMarkPoint> GetFaceMarkPoint(ViewFace viewFace, SKBitmap bitmap)
        {
            var infos = viewFace.FaceDetector(bitmap);
            var info = infos.First();
            return viewFace.FaceMark(bitmap, info).ToList();
        }
    }
}