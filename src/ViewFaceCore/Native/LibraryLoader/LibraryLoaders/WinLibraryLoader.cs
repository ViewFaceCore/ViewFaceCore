using System;
using System.Collections.Generic;
using System.Text;
using ViewFaceCore.Configs;
using ViewFaceCore.Exceptions;
using ViewFaceCore.Native.LibraryLoader.Interface;
using ViewFaceCore.Native.LibraryLoader.LibraryLoaders.Platforms;

namespace ViewFaceCore.Native.LibraryLoader.LibraryLoaders
{
    internal class WinLibraryLoader : BaseLibraryLoader
    {
        private readonly List<IntPtr> _ptrs = new List<IntPtr>();

        public override void Dispose()
        {
            throw new NotImplementedException();
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
                        string supportTennisLibPath = PathResolver.GetLibraryFullName("tennis_sandy_bridge");
                        if (!File.Exists(supportTennisLibPath))
                        {
                            return;
                        }
                        string baseTennisLibPath = PathResolver.GetLibraryFullName("tennis");
                        if (File.Exists(supportTennisLibPath))
                        {
                            File.Delete(baseTennisLibPath);
                        }
                        File.Copy(supportTennisLibPath, baseTennisLibPath, true);
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
                        string supportTennisLibPath = PathResolver.GetLibraryFullName("tennis_pentium");
                        if (!File.Exists(supportTennisLibPath))
                        {
                            return;
                        }
                        string baseTennisLibPath = PathResolver.GetLibraryFullName("tennis");
                        if (File.Exists(supportTennisLibPath))
                        {
                            File.Delete(baseTennisLibPath);
                        }
                        File.Copy(supportTennisLibPath, baseTennisLibPath, true);
                    }
                    break;
            }
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
        }

        protected override void Loading()
        {
            foreach (var library in BaseLibraryNames)
            {
                //Combine Library Path
                string libraryPath = PathResolver.GetLibraryFullName(library);
                if (!File.Exists(libraryPath))
                {
                    if (library.IndexOf("tennis_", StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        continue;
                    }
                    throw new FileNotFoundException($"Can not found library {libraryPath}.");
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
    }
}
