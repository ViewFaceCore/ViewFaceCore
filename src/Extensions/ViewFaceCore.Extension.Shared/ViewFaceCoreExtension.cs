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
            using (var faceImage = image.ToFaceImage())
            {
                foreach (var info in viewFace.FaceDetector(faceImage))
                {
                    yield return info;
                }
            }
        }

        public static IEnumerable<FaceMarkPoint> FaceMark<T>(this ViewFace viewFace, T image, FaceInfo info) where T : class
        {
            using (var faceImage = image.ToFaceImage())
            {
                foreach (var point in viewFace.FaceMark(faceImage, info))
                {
                    yield return point;
                }
            }
        }

    }
}
