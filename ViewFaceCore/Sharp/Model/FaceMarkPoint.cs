using System.Drawing;
using System.Runtime.InteropServices;

namespace ViewFaceCore.Sharp.Model
{
    /// <summary>
    /// 点坐标
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct FaceMarkPoint
    {
        private double x;
        private double y;

        /// <summary>
        /// 横坐标
        /// </summary>
        public double X { get => x; set => x = value; }
        /// <summary>
        /// 纵坐标
        /// </summary>
        public double Y { get => y; set => y = value; }

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
