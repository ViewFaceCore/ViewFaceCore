using System.Runtime.InteropServices;

namespace ViewFaceCore.Model
{
    /// <summary>
    /// 人脸跟踪信息
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct FaceTrackInfo
    {
        private readonly FaceRect pos;
        private readonly float score;

        private readonly int frame_no;
        private readonly int PID;
        private readonly int step;

        /// <summary>
        /// 人脸位置
        /// </summary>
        public FaceRect Location => pos;
        /// <summary>
        /// 人脸置信度
        /// </summary>
        public float Score => score;
        /// <summary>
        /// 人脸标识Id
        /// </summary>
        public int Pid => PID;
    }
}
