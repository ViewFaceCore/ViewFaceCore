namespace ViewFaceCore.Sharp.Model
{
    /// <summary>
    /// 面部信息
    /// </summary>
    public struct FaceInfo
    {
        private FaceRect location;
        private float score;

        /// <summary>
        /// 人脸位置
        /// </summary>
        public FaceRect Location { get => location; set => location = value; }
        /// <summary>
        /// 人脸置信度
        /// </summary>
        public float Score { get => score; set => score = value; }
    }
}
