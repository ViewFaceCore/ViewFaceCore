using System.Drawing;
using System.Runtime.InteropServices;

namespace ViewFaceCore.Sharp.Model
{
    /// <summary>
    /// 人脸位置矩形
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct FaceRect
    {
        private readonly int x;
        private readonly int y;
        private readonly int width;
        private readonly int height;

        /// <summary>
        /// 左上角点横坐标
        /// </summary>
        public int X => x;
        /// <summary>
        /// 左上角点纵坐标
        /// </summary>
        public int Y => y;
        /// <summary>
        /// 矩形宽度
        /// </summary>
        public int Width => width;
        /// <summary>
        /// 矩形高度
        /// </summary>
        public int Height => height;
    }
}
