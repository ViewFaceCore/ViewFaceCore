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
                if(options == null)
                {
                    throw new Exception("Can not get view face core options.");
                }
                //添加默认的能力
                AddViewFaceCore(services);
                if (options.IsEnableFaceAntiSpoofing)
                {
                    //活体检测
                    services.TryAddSingleton(new FaceAntiSpoofing());
                }

                return services;
            }
            finally
            {
                var exists = services.Where(p => p.ServiceType == typeof(IViewFaceFactory)).ToList();
                exists?.ForEach(e => { services.Remove(e); });
                services.TryAddSingleton<IViewFaceFactory>(new ViewFaceFactory(services));
            }
        }

        public static IServiceCollection AddViewFaceCore(this IServiceCollection services)
        {
            try
            {
                //人脸检测
                services.TryAddSingleton(new FaceDetector());
                //人脸标记
                services.TryAddSingleton(new FaceLandmarker());
                //人脸识别
                services.TryAddSingleton(new FaceRecognizer());
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
