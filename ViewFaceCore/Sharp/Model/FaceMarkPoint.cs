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
        /// .ctor
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        public FaceMarkPoint(double x, double y) : this()
        {
            this.x = x;
            this.y = y;
        }

        /// <summary>
        /// 横坐标
        /// </summary>
        public double X { get => x; set => x = value; }
        /// <summary>
        /// 纵坐标
        /// </summary>
        public double Y { get => y; set => y = value; }

        /// <summary>
        /// <see cref="FaceMarkPoint"/> to <see cref="PointF"/> 。
        /// </summary>
        /// <param name="point"></param>
        public static implicit operator PointF(FaceMarkPoint point)
        {
            return new PointF((float)point.X, (float)point.Y);
        }

        /// <summary>
        /// <see cref="PointF"/> to <see cref="FaceMarkPoint"/>。
        /// </summary>
        /// <param name="point"></param>
        public static implicit operator FaceMarkPoint(PointF point)
        {
            return new FaceMarkPoint(point.X, point.Y);
        }
    }}
