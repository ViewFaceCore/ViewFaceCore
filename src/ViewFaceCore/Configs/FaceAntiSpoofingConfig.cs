using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace ViewFaceCore.Configs
{
    /// <summary>
    /// 活体检测配置
    /// </summary>
    public class FaceAntiSpoofingConfig : BaseConfig
    {

        /// <summary>
        /// 设置视频帧数
        /// </summary>
        /// <remarks>
        /// 默认为10，当在PredictVideo模式下，输出帧数超过这个number之后，就可以输出识别结果。
        /// 这个数量相当于多帧识别结果融合的融合的帧数。
        /// 当输入的帧数超过设定帧数的时候，会采用滑动窗口的方式，返回融合的最近输入的帧融合的识别结果。
        /// 一般来说，在10以内，帧数越多，结果越稳定，相对性能越好，但是得到结果的延时越高。
        /// </remarks>
        public int VideoFrameCount { get; set; } = 10;

        /// <summary>
        /// 设置全局检测阈值
        /// </summary>
        /// <remarks>
        /// 默认为0.8，这个是攻击介质存在的分数阈值，该阈值越高，表示对攻击介质的要求越严格，一般的疑似就不会认为是攻击介质。这个一般不进行调整。
        /// </remarks>
        public float BoxThresh { get; set; } = 0.8f;

        /// <summary>
        /// 设置识别阈值
        /// </summary>
        /// <remarks>
        /// 默认为(0.3, 0.8)。
        /// 活体识别时，如果清晰度(clarity)低的话，就会直接返回FUZZY。清晰度满足阈值，则判断真实度（reality），超过阈值则认为是真人，低于阈值是攻击。
        /// 在视频识别模式下，会计算视频帧数内的平均值再跟帧数比较。两个阈值都符合，越高的话，越是严格。
        /// </remarks>
        public FaceAntiSpoofingConfigThreshold Threshold { get; set; } = new FaceAntiSpoofingConfigThreshold(0.3f, 0.8f);

        private bool _global = true;

        /// <summary>
        /// 是否开启全局检测模型，默认true
        /// </summary>
        /// <remarks>
        /// 活体检测识别器可以加载一个局部检测模型或者局部检测模型+全局检测模型。
        /// </remarks>
        public bool Global
        {
            get
            {
                return _global;
            }
            set
            {
#if !DEBUG
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && !value)
                {
                    throw new NotSupportedException("活体检测{局部检测模型}在Linux中存在问题，暂不支持在Linux中设置此选项为false");
                }

#endif
                _global = value;
            }
        }
    }

    public class FaceAntiSpoofingConfigThreshold
    {
        /// <summary>
        /// 设置识别阈值，默认为(0.3, 0.8)。
        /// </summary>
        /// <param name="clarity"></param>
        /// <param name="reality"></param>
        public FaceAntiSpoofingConfigThreshold(float clarity, float reality)
        {
            Clarity = clarity;
            Reality = reality;
        }

        /// <summary>
        /// 清晰度(clarity)
        /// </summary>
        public float Clarity { get; set; }

        /// <summary>
        /// 真实度（reality）
        /// </summary>
        public float Reality { get; set; }
    }
}
