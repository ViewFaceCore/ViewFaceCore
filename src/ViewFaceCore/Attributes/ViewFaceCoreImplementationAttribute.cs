using System;

namespace ViewFaceCore.Attributes;

/// <summary>
/// ViewFaceCore 的特定图形库实现
/// </summary>
[AttributeUsage(AttributeTargets.Assembly | AttributeTargets.Class, Inherited = false, AllowMultiple = true)]
internal sealed class ViewFaceCoreImplementationAttribute : Attribute
{
    readonly Type imageType;

    /// <summary>
    /// 待实现的图像类型
    /// </summary>
    public Type ImageType { get => imageType; }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="imageType">待实现的图像类型</param>
    public ViewFaceCoreImplementationAttribute(Type imageType)
    {
        this.imageType = imageType;
    }
}
