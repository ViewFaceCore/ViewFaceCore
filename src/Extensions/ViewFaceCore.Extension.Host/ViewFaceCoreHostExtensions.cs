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
        public static IServiceCollection AddViewFaceCore(this IServiceCollection services, Action<ViewFaceCoreOptions> option)
        {
            try
            {
                services.Configure("ViewFaceCoreOptions", option);

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

                if (options.IsEnableAll || options.IsEnableAgePredict)
                {
                    services.TryAddSingleton(new AgePredictor(options.AgePredictConfig));
                }

                if (options.IsEnableAll || options.IsEnableEyeStateDetect)
                {
                    services.TryAddSingleton(new EyeStateDetector(options.EyeStateDetectConfig));
                }

                if (options.IsEnableAll || options.IsEnableGenderPredict)
                {
                    services.TryAddSingleton(new GenderPredictor(options.GenderPredictConfig));
                }

                if (options.IsEnableAll || options.IsEnableFaceTrack)
                {
                    services.TryAddSingleton(new FaceTracker(options.FaceTrackerConfig));
                }

                if (options.IsEnableAll || options.IsEnablMaskDetect)
                {
                    services.TryAddSingleton(new MaskDetector(options.MaskDetectConfig));
                }

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
                return options.CurrentValue;
            }
        }
    }
}
