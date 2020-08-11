using System.Drawing;
using System.Runtime.InteropServices;

namespace ViewFaceCore.Sharp.Model
{
    /// <summary>
    /// 矩形
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct FaceRect
    {
        private int x;
        private int y;
        private int width;
        private int height;

        /// <summary>
        /// 左上角点横坐标
        /// </summary>
        public int X { get => x; set => x = value; }
        /// <summary>
        /// 左上角点纵坐标
        /// </summary>
        public int Y { get => y; set => y = value; }
        /// <summary>
        /// 矩形宽度
        /// </summary>
        public int Width { get => width; set => width = value; }
        /// <summary>
        /// 矩形高度
        /// </summary>
        public int Height { get => height; set => height = value; }

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
