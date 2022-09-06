#if !NET45_OR_GREATER // net40 »ò¸üµÍ°æ±¾

#if PORTABLE

namespace System.Threading
{
    internal class ExecutionContext
    {
        internal static ExecutionContext Capture()
        {
            return null;
        }

        internal static void Run(ExecutionContext executionContext, ContextCallback callback, object state)
        {
            callback(state);
        }
    }

    internal delegate void ContextCallback(object state);
}

#endif

#endif