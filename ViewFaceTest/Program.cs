using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using View.Core.Extensions;
using View.Drawing.Extensions;
using ViewFaceCore.Sharp;
using ViewFaceCore.Sharp.Configs;
using ViewFaceCore.Sharp.Model;

namespace ViewFaceTest
{
    class Program
    {
        static void Main()
        {
            // 老图片路径
            string oldImgPath = @"C:/Project/View.local.Packages/Images/Image.ViewFace.Test/5.jpg";
            string newImgPath = @"C:/Project/View.local.Packages/Images/Image.ViewFace.Test/1.jpg";

            ViewFace viewFace = new ViewFace((str) => { Debug.WriteLine(str); }); // 初始化人脸识别类，并设置 日志回调函数
            // 系统默认使用的轻量级识别模型。如果对精度有要求，请切换到 Normal 模式；并下载需要模型文件 放入生成目录的 model 文件夹中
            viewFace.FaceType = FaceType.Normal;
            // 系统默认使用5个人脸关键点。//不建议改动，除非是使用口罩模型。
            viewFace.MarkType = MarkType.Light;

            #region 识别老照片

            float[] oldEigenValues;
            Bitmap oldImg = (Bitmap)Image.FromFile(oldImgPath); // 从文件中加载照片 // 或者视频帧等

            var oldFaceInfos = viewFace.FaceDetector(oldImg); // 检测图片中包含的人脸信息。(置信度、位置、大小)
            oldFaceInfos = viewFace.FaceDetectorAsync(oldImg).GetAwaiter().GetResult(); // 异步检测图片中的人脸信息

            if (oldFaceInfos.Length > 0) //识别到人脸
            {
                { // 打印人脸信息
                    Console.WriteLine($"识别到的人脸数量：{oldFaceInfos.Length} 。人脸信息：\n");
                    Console.WriteLine($"序号\t人脸置信度\t位置X\t位置Y\t宽度\t高度");
                    for (int i = 0; i < oldFaceInfos.Length; i++)
                    {
                        Console.WriteLine($"{i + 1}\t{oldFaceInfos[i].Score:f8}\t{oldFaceInfos[i].Location.X}\t{oldFaceInfos[i].Location.Y}\t{oldFaceInfos[i].Location.Width}\t{oldFaceInfos[i].Location.Height}");
                    }
                    Console.WriteLine();
                }
                var oldPoints = viewFace.FaceMark(oldImg, oldFaceInfos[0]); // 获取 第一个人脸 的识别关键点。(人脸识别的关键点数据)
                oldEigenValues = viewFace.Extract(oldImg, oldPoints); // 获取 指定的关键点 的特征值。
                var state = viewFace.AntiSpoofing(oldImg, oldFaceInfos[0], oldPoints);
                Console.WriteLine($"单帧 活体检测结果：{state}");
                state = viewFace.AntiSpoofingVideo(oldImg, oldFaceInfos[0], oldPoints);
                Console.WriteLine($"视频 活体检测结果：{state}");

                Console.WriteLine();
                #region 质量评估
                var qualityBrightness = viewFace.FaceQuality(oldImg, oldFaceInfos[0], oldPoints, QualityType.Brightness);
                Console.WriteLine($"亮度:Level [{qualityBrightness.Level}] - Score [{qualityBrightness.Score}]");
                var qualityClarity = viewFace.FaceQuality(oldImg, oldFaceInfos[0], oldPoints, QualityType.Clarity);
                Console.WriteLine($"清晰度:Level [{qualityClarity.Level}] - Score [{qualityClarity.Score}]");
                var qualityIntegrity = viewFace.FaceQuality(oldImg, oldFaceInfos[0], oldPoints, QualityType.Integrity);
                Console.WriteLine($"完整度:Level [{qualityIntegrity.Level}] - Score [{qualityIntegrity.Score}]");
                var qualityPose = viewFace.FaceQuality(oldImg, oldFaceInfos[0], oldPoints, QualityType.Pose);
                Console.WriteLine($"姿态:Level [{qualityPose.Level}] - Score [{qualityPose.Score}]");
                var qualityPoseEx = viewFace.FaceQuality(oldImg, oldFaceInfos[0], oldPoints, QualityType.PoseEx);
                Console.WriteLine($"姿态 (深度):Level [{qualityPoseEx.Level}] - Score [{qualityPoseEx.Score}]");
                var qualityResolution = viewFace.FaceQuality(oldImg, oldFaceInfos[0], oldPoints, QualityType.Resolution);
                Console.WriteLine($"分辨率:Level [{qualityResolution.Level}] - Score [{qualityResolution.Score}]");
                var qualityStructure = viewFace.FaceQuality(oldImg, oldFaceInfos[0], oldPoints, QualityType.Structure);
                Console.WriteLine($"遮挡:Level [{qualityStructure.Level}] - Score [{qualityStructure.Score}]");
                var qualityClarityEx = viewFace.FaceQuality(oldImg, oldFaceInfos[0], oldPoints, QualityType.ClarityEx);
                Console.WriteLine($"清晰度 (深度):Level [{qualityClarityEx.Level}] - Score [{qualityClarityEx.Score}]");
                #endregion

                Console.WriteLine();
                #region 人脸属性
                var age = viewFace.FaceAgePredictor(oldImg, oldPoints);
                Console.WriteLine($"人脸预测年龄:{age}");
                var gender = viewFace.FaceGenderPredictor(oldImg, oldPoints);
                Console.WriteLine($"人脸预测性别:{gender.ToDescription()}");
                var eyeStateResult = viewFace.FaceEyeStateDetector(oldImg, oldPoints);
                Console.WriteLine($"左眼状态:{eyeStateResult.LeftEyeState.ToDescription()}");
                Console.WriteLine($"右眼状态:{eyeStateResult.RightEyeState.ToDescription()}");
                #endregion
            }
            else { oldEigenValues = new float[0]; /*未识别到人脸*/ }
            #endregion
            Console.WriteLine();
            #region 人脸追踪
            var trackFaceInfos = viewFace.FaceTrack(oldImg); // 人脸追踪
            if (trackFaceInfos.Length > 0)
            {
                { // 打印人脸信息
                    Console.WriteLine($"\r\n跟踪到的人脸数量：{trackFaceInfos.Length} 。人脸信息：\n");
                    Console.WriteLine($"PID\t人脸置信度\t位置X\t位置Y\t宽度\t高度");
                    for (int i = 0; i < trackFaceInfos.Length; i++)
                    {
                        Console.WriteLine($"{trackFaceInfos[i].Pid}\t{trackFaceInfos[i].Score:f8}\t{trackFaceInfos[i].Location.X}\t{trackFaceInfos[i].Location.Y}\t{trackFaceInfos[i].Location.Width}\t{trackFaceInfos[i].Location.Height}");
                    }
                }
            }
            #endregion
            Console.WriteLine();
            #region 识别新照片
            float[] newEigenValues;
            Bitmap newImg = (Bitmap)Image.FromFile(newImgPath/*新图片路径*/); // 从文件中加载照片 // 或者视频帧等
            newImg = (Bitmap)newImg.ChangeSize(new Size(1024, 768));
            var newFaces = viewFace.FaceDetector(newImg); // 检测图片中包含的人脸信息。(置信度、位置、大小)
            if (newFaces.Length > 0) //识别到人脸
            {
                { // 打印人脸信息
                    Console.WriteLine($"识别到的人脸数量：{newFaces.Length} 。人脸信息：\n");
                    Console.WriteLine($"序号\t人脸置信度\t位置X\t位置Y\t宽度\t高度");
                    for (int i = 0; i < newFaces.Length; i++)
                    {
                        Console.WriteLine($"{i + 1}\t{newFaces[i].Score:f8}\t{newFaces[i].Location.X}\t{newFaces[i].Location.Y}\t{newFaces[i].Location.Width}\t{newFaces[i].Location.Height}");
                    }
                }
                var newPoints = viewFace.FaceMark(newImg, newFaces[0]); // 获取 第一个人脸 的识别关键点。(人脸识别的关键点数据)
                newEigenValues = viewFace.Extract(newImg, newPoints); // 获取 指定的关键点 的特征值。
            }
            else { newEigenValues = new float[0]; /*未识别到人脸*/ }
            #endregion
            Console.WriteLine();
            try
            {
                float similarity = viewFace.Similarity(oldEigenValues, newEigenValues); // 对比两张照片上的数据，确认是否是同一个人。
                Console.WriteLine($"阈值 = {FaceCompareConfig.GetThreshold(viewFace.FaceType)}\t相似度 = {similarity}");
                Console.WriteLine($"是否是同一个人：{viewFace.IsSelf(similarity)}");
            }
            catch (Exception e)
            { Console.WriteLine(e.Message); }

            Console.ReadKey();
        }
    }
}
