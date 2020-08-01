using System;
using System.Collections.Generic;
using System.Text;
using ViewFaceCore.Sharp.Configs;

namespace ViewFaceCore.Sharp.Extensions
{
    /// <summary>
    /// <see cref="FaceDetectorConfig"/> 扩展。
    /// </summary>
    public static class FaceDetectorConfigExtension
    {
        /// <summary>
        /// 设置最小人脸。
        /// <para>
        /// 最小人脸是人脸检测器常用的一个概念，默认值为20，单位像素。<br />
        /// 最小人脸和检测器性能息息相关。主要方面是速度，使用建议上，我们建议在应用范围内，这个值设定的越大越好。<see langword="SeetaFace"/> 采用的是 <c><see langword="BindingBox Regresion"/></c> 的方式训练的检测器。
        /// 如果最小人脸参数设置为 <see langword="80"/> 的话，从检测能力上，可以将原图缩小的原来的 <see langword="1/4"/> ，这样从计算复杂度上，能够比最小人脸设置为 <see langword="20"/> 时，提速到 <see langword="16"/> 倍。
        /// </para>
        /// </summary>
        /// <param name="facesize"></param>
        /// <returns></returns>
        public static FaceDetectorConfig SetFaceSize(this FaceDetectorConfig faceDetector, double facesize)
        {
            faceDetector.FaceSize = facesize;
            return faceDetector;
        }
        /// <summary>
        /// 设置检测器阈值。
        /// <para>默认值是0.9，合理范围为[0, 1]。这个值一般不进行调整，除了用来处理一些极端情况。这个值设置的越小，漏检的概率越小，同时误检的概率会提高。</para>
        /// </summary>
        public static FaceDetectorConfig SetThreshold(this FaceDetectorConfig faceDetector, double threshold)
        {
            faceDetector.Threshold = threshold;
            return faceDetector;
        }
        /// <summary>
        /// 设置可检测的图像最大高度。
        /// <para>默认值2000。</para>
        /// </summary>
        public static FaceDetectorConfig SetMaxWidth(this FaceDetectorConfig faceDetector, double maxWidth)
        {
            faceDetector.MaxWidth = maxWidth;
            return faceDetector;
        }
        /// <summary>
        /// 设置可检测的图像最大高度。
        /// <para>默认值2000。</para>
        /// </summary>
        public static FaceDetectorConfig SetMaxHeight(this FaceDetectorConfig faceDetector, double maxHeight)
        {
            faceDetector.MaxHeight = maxHeight;
            return faceDetector;
        }
    }
}
