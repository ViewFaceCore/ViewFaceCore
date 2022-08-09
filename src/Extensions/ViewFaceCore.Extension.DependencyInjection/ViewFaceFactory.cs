using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using System;
using System.Collections.Generic;
using System.Text;
using ViewFaceCore.Core;

namespace ViewFaceCore.Extension.DependencyInjection
{
    public class ViewFaceFactory : IViewFaceFactory, IDisposable
    {
        private readonly ServiceProvider _provider;
        private readonly ViewFaceCoreOptions _options;
        private readonly Dictionary<Type, bool> _capabilityStatus;

        public ViewFaceFactory(IServiceCollection services)
        {
            if (services == null)
            {
                throw new ArgumentNullException(nameof(services));
            }
            _provider = services.BuildServiceProvider();
            //获取Options
            var options = _provider.GetRequiredService<IOptionsMonitor<ViewFaceCoreOptions>>();
            _options = options.Get(ViewFaceCoreOptions.OptionName) ?? throw new ArgumentNullException("Can not load ViewFaceCore options.");
            //构建基础能力
            _capabilityStatus = BuildCapabilityStatus();
        }

        public T Get<T>() where T : IViewFace
        {
            if (_capabilityStatus.TryGetValue(typeof(T), out bool status) && !status)
            {
                throw new NotSupportedException($"{typeof(T).Name} capability is not enabled, please enable it at first.");
            }
            return (T)_provider.GetService(typeof(T));
        }

        private Dictionary<Type, bool> BuildCapabilityStatus()
        {
            Dictionary<Type, bool> result = new Dictionary<Type, bool>()
            {
                //基础能力
                { typeof(FaceDetector),true},
                { typeof(FaceLandmarker),true},
                { typeof(FaceRecognizer),true},

                //非基础能力
                { typeof(FaceAntiSpoofing),(_options.IsEnableFaceAntiSpoofing || _options.IsEnableAll)},
                { typeof(AgePredictor),(_options.IsEnableAgePredict || _options.IsEnableAll)},
                { typeof(EyeStateDetector),(_options.IsEnableEyeStateDetect || _options.IsEnableAll)},
                { typeof(GenderPredictor),(_options.IsEnableGenderPredict || _options.IsEnableAll)},
                { typeof(FaceTracker),(_options.IsEnableFaceTrack || _options.IsEnableAll)},
                { typeof(MaskDetector),(_options.IsEnablMaskDetect || _options.IsEnableAll)},
                { typeof(FaceQuality),(_options.IsEnableQuality || _options.IsEnableAll)},
            };
            return result;
        }

        public void Dispose()
        {
            if (_provider != null)
            {
                _provider.Dispose();
            }
        }
    }
}
