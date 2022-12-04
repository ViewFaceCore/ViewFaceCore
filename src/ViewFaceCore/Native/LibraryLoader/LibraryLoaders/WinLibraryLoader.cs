using System;
using System.Collections.Generic;
using System.Text;
using ViewFaceCore.Configs;
using ViewFaceCore.Exceptions;
using ViewFaceCore.Native.LibraryLoader.Interface;
using ViewFaceCore.Native.LibraryLoader.LibraryLoaders.Platforms;
using System.IO;
using System.Runtime.InteropServices;

#if NETCOREAPP3_1_OR_GREATER
using System.Runtime.Intrinsics.X86;
#endif

namespace ViewFaceCore.Native.LibraryLoader.LibraryLoaders
{
    internal sealed class WinLibraryLoader : BaseLibraryLoader
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

#if NETCOREAPP3_1_OR_GREATER
            //不支持Avx2
            if (!Avx2.IsSupported)
            {
                if (BaseLibraryNames.Contains("tennis_haswell"))
                {
                    GlobalConfig.WriteLog("Detected that the CPU instruction does not support AVX2, disable tennis_haswell.");
                    BaseLibraryNames.Remove("tennis_haswell");
                }
                if (BaseLibraryNames.Contains("tennis_sandy_bridge"))
                {
                    GlobalConfig.WriteLog("Detected that the CPU instruction does not support AVX2, disable tennis_sandy_bridge.");
                    BaseLibraryNames.Remove("tennis_sandy_bridge");
                }
            };
            //不支持Fma
            if (!Fma.IsSupported)
            {
                if (BaseLibraryNames.Contains("tennis_sandy_bridge"))
                {
                    GlobalConfig.WriteLog("Detected that the CPU instruction does not support FMA, disable tennis_sandy_bridge.");
                    BaseLibraryNames.Remove("tennis_sandy_bridge");
                }
            };
#endif
        }

        protected override void SetModelsPath(string path)
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
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
            ViewFaceNative.SetModelPathWindows(Encoding.UTF8.GetString(pathUtf8Bytes));
            if (!path.Equals(ViewFaceNative.GetModelPath()))
            {
                throw new LoadModelException($"Set model path to '{path}' failed, failed to verify this path.");
            }
            GlobalConfig.WriteLog($"Loading models from {path}");
        }

        protected override void Loading()
        {
            GlobalConfig.WriteLog($"Loading library from {PathResolver.GetLibraryPath()}");

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
                    LoadViewFaceBridge(libraryPath);
                    continue;
                }

#if NETCOREAPP3_1_OR_GREATER
                IntPtr ptr = NativeLibrary.Load(libraryPath);
                if (ptr == IntPtr.Zero)
                {
                    throw new BadImageFormatException($"Can not load native library {libraryPath}.");
                }
#else
                IntPtr ptr = Kernel32.LoadLibrary(libraryPath);
                if (ptr == IntPtr.Zero)
                {
                    throw new BadImageFormatException($"Can not load native library {libraryPath}.");
                }
#endif
                _ptrs.Add(ptr);
            }
        }

        private void LoadViewFaceBridge(string libraryPath)
        {
#if NETCOREAPP3_1_OR_GREATER
            NativeLibrary.SetDllImportResolver(Assembly.GetAssembly(typeof(ViewFaceNative)), (libraryName, assembly, searchPath) =>
            {
                return NativeLibrary.Load(libraryPath, assembly, searchPath ?? DllImportSearchPath.UseDllDirectoryForDependencies);
            });
#else
            IntPtr ptr = Kernel32.LoadLibrary(libraryPath);
            if (ptr == IntPtr.Zero)
            {
                throw new BadImageFormatException($"Can not load native library {libraryPath}.");
            }
#endif
        }
    }
}