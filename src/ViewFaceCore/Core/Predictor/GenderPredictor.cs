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
    /// 性别预测。
    /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.gender_predictor">gender_predictor.csta</a>
    /// </summary>
    public sealed class GenderPredictor : BaseViewFace, IPredictor
    {
        private readonly IntPtr _handle = IntPtr.Zero;
        private readonly static object _locker = new object();
        public GenderPredictConfig GenderPredictConfig { get; private set; }

        public GenderPredictor(GenderPredictConfig config = null)
        {
            this.GenderPredictConfig = config ?? new GenderPredictConfig();
            _handle = ViewFaceNative.GetGenderPredictorHandler((int)this.GenderPredictConfig.DeviceType);
            if (_handle == IntPtr.Zero)
            {
                throw new Exception("Get gender predictor handler failed.");
            }
        }

        /// <summary>
        /// 性别预测。
        /// <para>
        /// 需要模型 <a href="https://www.nuget.org/packages/ViewFaceCore.model.gender_predictor">gender_predictor.csta</a>
        /// </para>
        /// </summary>
        /// <param name="image">人脸图像信息</param>
        /// <param name="points">关键点坐标<para>通过 <see cref="FaceMark(FaceImage, FaceInfo)"/> 获取</para></param>
        /// <returns>性别。<see cref="Gender.Unknown"/> 代表识别失败</returns>
        public Gender PredictGender(FaceImage image, FaceMarkPoint[] points)
        {
            lock (_locker)
            {
                int result = ViewFaceNative.PredictGender(_handle, ref image, points);
                if (Enum.TryParse(result.ToString(), out Gender gender))
                {
                    return gender;
                }
                return Gender.Unknown;
            }
        }

        public void Dispose()
        {
            lock (_locker)
            {
                ViewFaceNative.DisposeGenderPredictor(_handle);
            }
        }
    }
}
