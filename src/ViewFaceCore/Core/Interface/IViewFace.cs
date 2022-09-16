namespace ViewFaceCore.Core;

/// <summary>
/// ViewFace Interface
/// </summary>
public interface IViewFace : IDisposable
{
    /// <summary>
    /// 获取模型路径
    /// </summary>
    public string ModelPath { get; }

    /// <summary>
    /// 获取库路径
    /// </summary>
    public string LibraryPath { get; }
}
