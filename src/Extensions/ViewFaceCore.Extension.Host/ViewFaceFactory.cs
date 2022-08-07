using Microsoft.Extensions.DependencyInjection;
using System;
using System.Collections.Generic;
using System.Text;
using ViewFaceCore.Core;

namespace ViewFaceCore.Extension.DependencyInjection
{
    public class ViewFaceFactory : IViewFaceFactory, IDisposable
    {
        private readonly ServiceProvider _provider;

        public ViewFaceFactory(IServiceCollection services)
        {
            if (services == null)
            {
                throw new ArgumentNullException(nameof(services));
            }
            _provider = services.BuildServiceProvider();
        }

        public T Get<T>() where T : IViewFace
        {
            return (T)_provider.GetService(typeof(T));
        }

        public void Dispose()
        {
            if(_provider != null)
            {
                _provider.Dispose();
            }
        }
    }
}
