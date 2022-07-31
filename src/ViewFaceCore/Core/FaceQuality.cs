using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ViewFaceCore.Configs;
using ViewFaceCore.Model;
using ViewFaceCore.Native;

namespace ViewFaceCore.Core
{
    /// <summary>
    /// 质量评估
    /// </summary>
    public sealed class FaceQuality : BaseViewFace, IDisposable
    {
        private readonly QualityConfig _qualityConfig = null;

        public FaceQuality(QualityConfig qualityConfig = null)
        {
            _qualityConfig = qualityConfig ?? new QualityConfig();
        }

        /// <summary>
        /// 人脸质量评估
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="info">面部信息<para>通过 <see cref="FaceDetector(FaceImage)"/> 获取</para></param>
        /// <param name="points"><paramref name="info"/> 对应的关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <param name="type">质量评估类型</param>
        /// <returns></returns>
        public QualityResult Detect(FaceImage image, FaceInfo info, FaceMarkPoint[] points, QualityType type)
        {
            int level = -1;
            float score = -1;

            switch (type)
            {
                case QualityType.Brightness:
                    ViewFaceNative.QualityOfBrightness(ref image, info.Location, points, points.Length, ref level, ref score,
                        _qualityConfig.Brightness.V0, _qualityConfig.Brightness.V1, _qualityConfig.Brightness.V2, _qualityConfig.Brightness.V3);
                    break;
                case QualityType.Clarity:
                    ViewFaceNative.QualityOfClarity(ref image, info.Location, points, points.Length, ref level, ref score, _qualityConfig.Clarity.Low, _qualityConfig.Clarity.High);
                    break;
                case QualityType.Integrity:
                    ViewFaceNative.QualityOfIntegrity(ref image, info.Location, points, points.Length, ref level, ref score,
                        _qualityConfig.Integrity.Low, _qualityConfig.Integrity.High);
                    break;
                case QualityType.Pose:
                    ViewFaceNative.QualityOfPose(ref image, info.Location, points, points.Length, ref level, ref score);
                    break;
                case QualityType.PoseEx:
                    ViewFaceNative.QualityOfPoseEx(ref image, info.Location, points, points.Length, ref level, ref score,
                       _qualityConfig.PoseEx.YawLow, _qualityConfig.PoseEx.YawHigh,
                       _qualityConfig.PoseEx.PitchLow, _qualityConfig.PoseEx.PitchHigh,
                       _qualityConfig.PoseEx.RollLow, _qualityConfig.PoseEx.RollHigh);
                    break;
                case QualityType.Resolution:
                    ViewFaceNative.QualityOfResolution(ref image, info.Location, points, points.Length, ref level, ref score, _qualityConfig.Resolution.Low, _qualityConfig.Resolution.High);
                    break;
                case QualityType.ClarityEx:
                    ViewFaceNative.QualityOfClarityEx(ref image, info.Location, points, points.Length, ref level, ref score, _qualityConfig.ClarityEx.BlurThresh);
                    break;
                case QualityType.Structure:
                    ViewFaceNative.QualityOfNoMask(ref image, info.Location, points, points.Length, ref level, ref score);
                    break;
            }

            return new QualityResult() { Level = (QualityLevel)level, Score = score };
        }

        public void Dispose()
        {

        }
    }
}
