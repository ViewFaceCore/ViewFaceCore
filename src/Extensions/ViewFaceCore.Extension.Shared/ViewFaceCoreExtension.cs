using System;
using System.Collections.Generic;
using System.Text;
using ViewFaceCore.Model;

namespace ViewFaceCore
{
    public static class ViewFaceCoreExtension
    {
        public static IEnumerable<FaceInfo> FaceDetector<T>(this ViewFace viewFace, T image) where T : class
        {
            var img = image.ToFaceImage();
            return viewFace.FaceDetector(img);
        }

    }
}
