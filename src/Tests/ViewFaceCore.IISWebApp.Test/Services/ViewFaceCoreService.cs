using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using ViewFaceCore.Core;

namespace ViewFaceCore.IISWebApp.Test.Services
{
    public class ViewFaceCoreService : IDisposable
    {
        public static ViewFaceCoreService Instance = new ViewFaceCoreService();

        public FaceDetector FaceDetector { get; private set; }

        private ViewFaceCoreService()
        {
            FaceDetector = new FaceDetector();
        }

        public void Dispose()
        {
            if(this.FaceDetector != null)
            {
                this.FaceDetector.Dispose();
            }
        }
    }
}