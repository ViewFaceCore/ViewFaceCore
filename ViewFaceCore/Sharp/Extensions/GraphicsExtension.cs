using System.Drawing;

namespace ViewFaceCore.Sharp.Extensions
{
    /// <summary>
    /// <see cref="Graphics"/> 扩展方法。
    /// </summary>
    public static class GraphicsExtension
    {
        /// <summary>
        /// 可链式调用的 <see cref="Graphics.DrawImage(Image, Rectangle, Rectangle, GraphicsUnit)"/>。
        /// </summary>
        /// <param name="graphics"></param>
        /// <param name="image"></param>
        /// <param name="destRect"></param>
        /// <param name="srcRect"></param>
        /// <param name="srcUnit"></param>
        /// <returns></returns>
        public static  Graphics DrawImageEx(this Graphics graphics, Image image, Rectangle destRect, Rectangle srcRect, GraphicsUnit srcUnit)
        {
            graphics.DrawImage(image, destRect, srcRect, srcUnit);
            return graphics;
        }
    }
}
