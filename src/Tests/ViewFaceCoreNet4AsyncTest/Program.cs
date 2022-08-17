using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ViewFaceCore;
using ViewFaceCore.Core;
using ViewFaceCore.Model;

namespace ViewFaceCoreNet4AsyncTest
{
    internal class Program
    {
        private readonly static string imagePath = @"images/Jay_3.jpg";
        private readonly static string imagePath1 = @"images/Jay_4.jpg";

        static void Main(string[] args)
        {
            FaceDetectorDemo().GetAwaiter().GetResult();

            Console.ReadKey(false);
        }

        static async Task FaceDetectorDemo()
        {
            using (var bitmap = (Bitmap)Image.FromFile(imagePath))
            {
                using (FaceDetector faceDetector = new FaceDetector())
                {
                    FaceInfo[] infos = await faceDetector.DetectAsync(bitmap);
                    Console.WriteLine($"识别到的人脸数量：{infos.Length} 个人脸信息：\n");
                    Console.WriteLine($"No.\t人脸置信度\t位置信息");
                    for (int i = 0; i < infos.Length; i++)
                    {
                        Console.WriteLine($"{i}\t{infos[i].Score:f8}\t{infos[i].Location}");
                    }
                }
            }
            Console.WriteLine();
        }
    }
}
