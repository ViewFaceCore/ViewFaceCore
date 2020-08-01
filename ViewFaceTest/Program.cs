using System;
using System.Diagnostics;
using System.Drawing;
using ViewFaceCore.Sharp;
using ViewFaceCore.Sharp.Configs;
using ViewFaceCore.Sharp.Model;

namespace ViewFaceTest
{
    class Program
    {
        static void Main()
        {
            ViewFace viewFace = new ViewFace((str) => { Debug.WriteLine(str); }); // 初始化人脸识别类，并设置 日志回调函数
            viewFace.DetectorSetting = new FaceDetectorConfig() { FaceSize = 20, MaxWidth = 2000, MaxHeight = 2000, Threshold = 0.5 };

            // 系统默认使用的轻量级识别模型。如果对精度有要求，请切换到 Normal 模式；并下载需要模型文件 放入生成目录的 model 文件夹中
            viewFace.FaceType = FaceType.Normal;
            // 系统默认使用5个人脸关键点。//不建议改动，除非是使用口罩模型。
            viewFace.MarkType = MarkType.Light;

            #region 识别老照片
            float[] oldEigenValues;
            Bitmap oldImg = (Bitmap)Image.FromFile(@"C:/Users/yangw/OneDrive/桌面/image/董建华.jpg"/*老图片路径*/); // 从文件中加载照片 // 或者视频帧等
            var oldFaces = viewFace.FaceDetector(oldImg); // 检测图片中包含的人脸信息。(置信度、位置、大小)
            if (oldFaces.Length > 0) //识别到人脸
            {
                { // 打印人脸信息
                    Console.WriteLine($"识别到的人脸数量：{oldFaces.Length} 。人脸信息：\n");
                    Console.WriteLine($"序号\t人脸置信度\t位置X\t位置Y\t宽度\t高度");
                    for (int i = 0; i < oldFaces.Length; i++)
                    {
                        Console.WriteLine($"{i + 1}\t{oldFaces[i].Score}\t{oldFaces[i].Location.X}\t{oldFaces[i].Location.Y}\t{oldFaces[i].Location.Width}\t{oldFaces[i].Location.Height}");
                    }
                    Console.WriteLine();
                }
                var oldPoints = viewFace.FaceMark(oldImg, oldFaces[0]); // 获取 第一个人脸 的识别关键点。(人脸识别的关键点数据)
                oldEigenValues = viewFace.Extract(oldImg, oldPoints); // 获取 指定的关键点 的特征值。
            }
            else { oldEigenValues = new float[0]; /*未识别到人脸*/ }
            var trackFaces = viewFace.FaceTrack(oldImg);
            #endregion

            #region 识别新照片
            float[] newEigenValues;
            Bitmap newImg = (Bitmap)Image.FromFile(@"C:/Users/yangw/OneDrive/桌面/image/董丽华.jpg"/*新图片路径*/); // 从文件中加载照片 // 或者视频帧等
            var newFaces = viewFace.FaceDetector(newImg); // 检测图片中包含的人脸信息。(置信度、位置、大小)
            if (newFaces.Length > 0) //识别到人脸
            {
                { // 打印人脸信息
                    Console.WriteLine($"识别到的人脸数量：{newFaces.Length} 。人脸信息：\n");
                    Console.WriteLine($"序号\t人脸置信度\t位置X\t位置Y\t宽度\t高度");
                    for (int i = 0; i < newFaces.Length; i++)
                    {
                        Console.WriteLine($"{i + 1}\t{newFaces[i].Score}\t{newFaces[i].Location.X}\t{newFaces[i].Location.Y}\t{newFaces[i].Location.Width}\t{newFaces[i].Location.Height}");
                    }
                    Console.WriteLine();
                }
                var newPoints = viewFace.FaceMark(newImg, newFaces[0]); // 获取 第一个人脸 的识别关键点。(人脸识别的关键点数据)
                newEigenValues = viewFace.Extract(newImg, newPoints); // 获取 指定的关键点 的特征值。
            }
            else { newEigenValues = new float[0]; /*未识别到人脸*/ }
            #endregion

            try
            {
                float similarity = viewFace.Similarity(oldEigenValues, newEigenValues); // 对比两张照片上的数据，确认是否是同一个人。
                Console.WriteLine($"阈值 = {FaceCompareConfig.GetThreshold(viewFace.FaceType)}\t相似度 = {similarity}");
                Console.WriteLine($"是否是同一个人：{viewFace.IsSelf(similarity)}");
            }
            catch (Exception e)
            { Console.WriteLine(e); }

            Console.ReadKey();
        }
    }
}
