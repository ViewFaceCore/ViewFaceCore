namespace ViewFaceCore.Sharp.Model
{
    /// <summary>
    /// 人脸跟踪信息
    /// </summary>
    public struct FaceTrackInfo
    {
        /// <summary>
        /// 人脸位置
        /// </summary>
        public FaceRect Location { get; set; }
        /// <summary>
        /// 人脸置信度
        /// </summary>
        public float Score { get; set; }
        /// <summary>
        /// 人脸标识Id
        /// </summary>
        public int Pid { get; set; }
    }
}
