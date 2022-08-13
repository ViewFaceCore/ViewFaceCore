using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ViewFaceCore.Model;

namespace ViewFaceCore.Extensions
{
    /// <summary>
    /// 人脸追踪结果扩展
    /// </summary>
    public static class FaceTrackInfoExtension
    {
        /// <summary>
        /// FaceTrackInfo转换为FaceInfo
        /// </summary>
        /// <param name="faceTrackInfo"></param>
        /// <returns></returns>
        public static FaceInfo ToFaceInfo(this FaceTrackInfo faceTrackInfo)
        {
            return new FaceInfo()
            {
                Score = faceTrackInfo.Score,
                Location = faceTrackInfo.Location,
            };
        }
    }
}
