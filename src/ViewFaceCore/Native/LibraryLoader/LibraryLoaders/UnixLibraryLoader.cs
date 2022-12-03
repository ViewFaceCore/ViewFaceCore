﻿using ViewFaceCore.Configs;
using ViewFaceCore.Exceptions;
using ViewFaceCore.Native.LibraryLoader.Interface;
using ViewFaceCore.Native.LibraryLoader.LibraryLoaders.Platforms;

namespace ViewFaceCore.Native.LibraryLoader.LibraryLoaders
{
    internal sealed class UnixLibraryLoader : BaseLibraryLoader
    {
        private readonly List<IntPtr> _ptrs = new List<IntPtr>();

        public override void Dispose()
        {
            if (_ptrs?.Any() != true)
            {
                return;
            }
            foreach (var item in _ptrs)
            {
                try
                {
#if NETCOREAPP3_OR_GREATER
                    NativeLibrary.Free(item);
#endif
                }
                catch { }
            }
        }

        protected override void SetInstructionSupport()
        {
            //Arm不需要处理
            if (RuntimeInformation.ProcessArchitecture != Architecture.X86
                && RuntimeInformation.ProcessArchitecture != Architecture.X64)
            {
                return;
            }
            GlobalConfig.WriteLog($"Instruction set to {GlobalConfig.X86Instruction}");
            switch (GlobalConfig.X86Instruction)
            {
                case X86Instruction.AVX2:
                    {
                        //只支持AVX2
                        GlobalConfig.WriteLog("CPU only support AVX2 instruction, will use tennis_sandy_bridge.");

                        List<string> removeLibs = new List<string>() { "tennis_haswell", "tennis_pentium" };
                        removeLibs.ForEach(p =>
                        {
                            if (BaseLibraryNames.Contains(p))
                            {
                                BaseLibraryNames.Remove(p);
                            }
                        });
                    }
                    break;
                case X86Instruction.SSE2:
                    {
                        //只支持SSE2
                        GlobalConfig.WriteLog("CPU only support SSE2 instruction, will use tennis_pentium.");

                        List<string> removeLibs = new List<string>() { "tennis_haswell", "tennis_sandy_bridge" };
                        removeLibs.ForEach(p =>
                        {
                            if (BaseLibraryNames.Contains(p))
                            {
                                BaseLibraryNames.Remove(p);
                            }
                        });
                    }
                    break;
            }
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
                throw new NotSupportedException(
                    $"The path is too long, not support path more than {ViewFaceNative.MAX_PATH_LENGTH} byte.");
            }

            ViewFaceNative.SetModelPathLinux(Encoding.UTF8.GetBytes(Encoding.UTF8.GetString(pathUtf8Bytes)));
            if (!path.Equals(ViewFaceNative.GetModelPath()))
            {
                throw new LoadModelException($"Set model path to '{path}' failed, failed to verify this path.");
            }
        }

        protected override void Loading()
        {
#if NETCOREAPP3_OR_GREATER
            foreach (var library in BaseLibraryNames)
            {
                string libraryPath = PathResolver.GetLibraryFullName(library);
                if (!File.Exists(libraryPath))
                {
                    if (library.IndexOf("tennis_", StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        continue;
                    }
                    throw new FileNotFoundException($"Can not found library {libraryPath}.");
                }

                if (library.IndexOf(ViewFaceNative.BRIDGE_LIBRARY_NAME, StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    NativeLibrary.SetDllImportResolver(Assembly.GetAssembly(typeof(ViewFaceNative)), (libraryName, assembly, searchPath) =>
                    {
                        return NativeLibrary.Load(libraryPath, assembly, searchPath ?? DllImportSearchPath.UseDllDirectoryForDependencies);
                    });
                    continue;
                }

                IntPtr ptr = NativeLibrary.Load(libraryPath);
                if (ptr == IntPtr.Zero)
                {
                    throw new BadImageFormatException($"Can not load native library {libraryPath}.");
                }
                _ptrs.Add(ptr);
            }
#else
            throw new NotSupportedException("On Linux, only .net core 3.1 and above are supported");
#endif
        }
    }
}