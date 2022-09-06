#if !NET45_OR_GREATER // net40 »ò¸üµÍ°æ±¾

namespace System.Runtime.CompilerServices
{
    /// <summary>
    /// Allows you to obtain the line number in the source file at which the method is called.
    /// </summary>
    [AttributeUsage(AttributeTargets.Parameter, Inherited = false)]
    public sealed class CallerLineNumberAttribute : Attribute
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CallerLineNumberAttribute"/> class.
        /// </summary>
        [TargetedPatchingOptOut("Performance critical to inline this type of method across NGen image boundaries")]
        public CallerLineNumberAttribute()
        {
        }
    }
}

#endif