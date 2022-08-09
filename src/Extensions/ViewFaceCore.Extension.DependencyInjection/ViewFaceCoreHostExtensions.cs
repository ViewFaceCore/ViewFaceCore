using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ViewFaceCore.Core;

namespace ViewFaceCore.Extension.DependencyInjection
{
    public static class ViewFaceCoreHostExtensions
    {
        public static IServiceCollection AddViewFaceCore(this IServiceCollection services, Action<ViewFaceCoreOptions> option = null)
        {
            try
            {
                if(option == null)
                {
                    option = (o) =>{ };
                }
                services.Configure(ViewFaceCoreOptions.OptionName, option);
                var options = GetOptions(services);
                if (options == null)
                {
                    throw new Exception("Can not get view face core options.");
                }

                //添加默认的能力
                //人脸检测
                services.TryAddSingleton(new FaceDetector(options.FaceDetectConfig));
                //人脸标记
                services.TryAddSingleton(new FaceLandmarker(options.FaceLandmarkConfig));
                //人脸识别
                services.TryAddSingleton(new FaceRecognizer(options.FaceRecognizeConfig));

                //活体检测
                if (options.IsEnableAll || options.IsEnableFaceAntiSpoofing)
                {
                    services.TryAddSingleton(new FaceAntiSpoofing(options.FaceAntiSpoofingConfig));
                }
                //年龄与猜测
                if (options.IsEnableAll || options.IsEnableAgePredict)
                {
                    services.TryAddSingleton(new AgePredictor(options.AgePredictConfig));
                }
                //眼睛状态检测
                if (options.IsEnableAll || options.IsEnableEyeStateDetect)
                {
                    services.TryAddSingleton(new EyeStateDetector(options.EyeStateDetectConfig));
                }
                //性别预测
                if (options.IsEnableAll || options.IsEnableGenderPredict)
                {
                    services.TryAddSingleton(new GenderPredictor(options.GenderPredictConfig));
                }
                //人脸追踪
                if (options.IsEnableAll || options.IsEnableFaceTrack)
                {
                    services.TryAddSingleton(new FaceTracker(options.FaceTrackerConfig));
                }
                //口罩识别
                if (options.IsEnableAll || options.IsEnablMaskDetect)
                {
                    services.TryAddSingleton(new MaskDetector(options.MaskDetectConfig));
                }
                //质量检测
                if (options.IsEnableAll || options.IsEnableQuality)
                {
                    services.TryAddSingleton(new FaceQuality(options.QualityConfig));
                }
                return services;
            }
            finally
            {
                services.TryAddSingleton<IViewFaceFactory>(new ViewFaceFactory(services));
            }
        }

        private static ViewFaceCoreOptions GetOptions(IServiceCollection services)
        {
            using (var provider = services.BuildServiceProvider())
            {
                var options = provider.GetRequiredService<IOptionsMonitor<ViewFaceCoreOptions>>();
                return options.Get(ViewFaceCoreOptions.OptionName);
            }
        }
    }
}
