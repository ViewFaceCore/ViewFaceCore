using ViewFaceCore.Exceptions;
using ViewFaceCore.Native.LibraryLoader.Interface;

namespace ViewFaceCore.Native.LibraryLoader.LibraryLoaders
{
    internal class UnixLibraryLoader : BaseLibraryLoader
    {
        public override void Dispose()
        {

        }

        protected override void SetInstructionSupport()
        {

        }

        protected override void SetModelsPath(string path)
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                throw new PlatformNotSupportedException($"Unsupported system type: {RuntimeInformation.OSDescription}");
            }
            if (string.IsNullOrWhiteSpace(path))
            {
                throw new ArgumentNullException(nameof(path), "Model path can not null.");
            }
            byte[] pathUtf8Bytes = Encoding.Convert(Encoding.Default, Encoding.UTF8, Encoding.Default.GetBytes(path));
            if (pathUtf8Bytes.Length > ViewFaceNative.MAX_PATH_LENGTH)
            {
                throw new NotSupportedException($"The path is too long, not support path more than {ViewFaceNative.MAX_PATH_LENGTH} byte.");
            }
            ViewFaceNative.SetModelPathLinux(Encoding.UTF8.GetBytes(Encoding.UTF8.GetString(pathUtf8Bytes)));
            if (!path.Equals(ViewFaceNative.GetModelPath()))
            {
                throw new SetModelException($"Set model path to '{path}' failed, failed to verify this path.");
            }
        }

        protected override void Loading()
        {

        }
    }
}
