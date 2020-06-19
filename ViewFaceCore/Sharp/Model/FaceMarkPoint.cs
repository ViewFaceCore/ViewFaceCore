using System.Drawing;

namespace ViewFaceCore.Sharp.Model
{
    /// <summary>
    /// 点坐标
    /// </summary>
    public struct FaceMarkPoint
    {
        /// <summary>
        /// 横坐标
        /// </summary>
        public double X { get; set; }
        /// <summary>
        /// 纵坐标
        /// </summary>
        public double Y { get; set; }

        /// <summary>
        /// <see cref="FaceMarkPoint"/> 到 <see cref="PointF"/> 的隐式转换。
        /// </summary>
        /// <param name="point"></param>
        public static implicit operator PointF(FaceMarkPoint point)
        {
            return new PointF((float)point.X, (float)point.Y);
        }
    };
}
