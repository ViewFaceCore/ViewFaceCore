using System.Drawing;

namespace ViewFaceCore.Sharp.Model
{
    /// <summary>
    /// 矩形
    /// </summary>
    public struct FaceRect
    {
        /// <summary>
        /// 左上角点横坐标
        /// </summary>
        public int X { get; set; }
        /// <summary>
        /// 左上角点纵坐标
        /// </summary>
        public int Y { get; set; }
        /// <summary>
        /// 矩形宽度
        /// </summary>
        public int Width { get; set; }
        /// <summary>
        /// 矩形高度
        /// </summary>
        public int Height { get; set; }

        /// <summary>
        /// <see cref="FaceRect"/> 到 <see cref="Rectangle"/> 的隐式转换。
        /// </summary>
        /// <param name="rect"></param>
        public static implicit operator Rectangle(FaceRect rect)
        {
            return new Rectangle(rect.X, rect.Y, rect.Width, rect.Height);
        }
    }
}
