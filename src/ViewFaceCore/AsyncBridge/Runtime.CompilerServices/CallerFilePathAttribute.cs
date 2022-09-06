#if !NET45_OR_GREATER // net40 »ò¸üµÍ°æ±¾

namespace System.Runtime.CompilerServices
{
    /// <summary>
    /// Allows you to obtain the full path of the source file that contains the caller. This is the file path at the time of compile.
    /// </summary>
    [AttributeUsage(AttributeTargets.Parameter, Inherited = false)]
    public sealed class CallerFilePathAttribute : Attribute
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CallerFilePathAttribute"/> class.
        /// </summary>
        [TargetedPatchingOptOut("Performance critical to inline this type of method across NGen image boundaries")]
        public CallerFilePathAttribute()
        {
        }
    }
}

#endif