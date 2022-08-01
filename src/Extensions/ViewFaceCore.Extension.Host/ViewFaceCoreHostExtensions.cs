using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using System;
using System.Collections.Generic;
using System.Text;
using ViewFaceCore.Core;

namespace ViewFaceCore.Extension.Host
{
    public static class ViewFaceCoreHostExtensions
    {
        public static IServiceCollection AddViewFaceCore(this IServiceCollection services, Action<ViewFaceCoreOptions> option)
        {
            services.Configure("ViewFaceCoreOptions", option);
            //添加默认的能力
            AddViewFaceCore(services);

            return services;
        }

        public static IServiceCollection AddViewFaceCore(this IServiceCollection services)
        {
            //人脸检测
            services.TryAddSingleton<FaceDetector>(new FaceDetector());
            //人脸标记
            services.TryAddSingleton(new FaceMark());
            //人脸识别
            services.TryAddSingleton(new FaceRecognizer());
            return services;
        }
    }
}
