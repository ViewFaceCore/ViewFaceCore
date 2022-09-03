using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace ViewFaceCore.Configs
{
    public static class GlobalConfig
    {
        #region Log

        public static Action<string> LogEvent { get; private set; }

        /// <summary>
        /// 绑定log event
        /// </summary>
        /// <param name="action"></param>
        public static void SetLog(Action<string> action)
        {
            if (action == null) return;
            if (LogEvent != null) return;
            LogEvent = action;
        }

        /// <summary>
        /// 对外写日志
        /// </summary>
        /// <param name="log"></param>
        internal static void WriteLog(string log)
        {
            if (LogEvent != null)
            {
                LogEvent(log);
            }
        }

        #endregion

        #region Instruction

        private static bool _isSetX86Instruction = false;

        private static readonly object _setX86InstructionLocker = new object();

        /// <summary>
        /// 在x86环境下支持的指令集
        /// </summary>
        public static X86Instruction X86Instruction { get; private set; } = X86Instruction.AVX2 | X86Instruction.SSE2 | X86Instruction.FMA;

        /// <summary>
        /// 设置支持的指令集
        /// </summary>
        /// <param name="instruction"></param>
        public static void SetX86Instruction(X86Instruction instruction)
        {
            if (_isSetX86Instruction)
                return;
            if (RuntimeInformation.ProcessArchitecture != Architecture.X86 && RuntimeInformation.ProcessArchitecture != Architecture.X64)
                return;
            lock (_setX86InstructionLocker)
            {
                if (_isSetX86Instruction)
                    return;
                _isSetX86Instruction = true;
                X86Instruction = instruction;
            }
        }

        #endregion
    }

    public enum X86Instruction
    {
        AVX2 = 1 << 0,

        SSE2 = 1 << 1,
        
        FMA = 1 << 2,
    }
}
