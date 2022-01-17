using System.Drawing;
using System.Runtime.InteropServices;

namespace ViewFaceCore.Model
{
    /// <summary>
    /// 点坐标
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct FaceMarkPoint
    {
        private readonly double x;
        private readonly double y;

        /// <summary>
        /// 横坐标
        /// </summary>
        public double X => x;
        /// <summary>
        /// 纵坐标
        /// </summary>
        public double Y => y;
    }
}
