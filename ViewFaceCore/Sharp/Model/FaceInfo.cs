namespace ViewFaceCore.Sharp.Model
{
    /// <summary>
    /// 面部信息
    /// </summary>
    public struct FaceInfo
    {
        /// <summary>
        /// 人脸位置
        /// </summary>
        public FaceRect Location { get; set; }
        /// <summary>
        /// 人脸置信度
        /// </summary>
        public float Score { get; set; }
    }
}
