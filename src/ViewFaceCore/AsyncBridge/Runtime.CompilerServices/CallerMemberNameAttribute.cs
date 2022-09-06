#if !NET45_OR_GREATER // net40 »ò¸üµÍ°æ±¾

namespace System.Runtime.CompilerServices
{
    /// <summary>
    /// Allows you to obtain the method or property name of the caller to the method.
    /// </summary>
    [AttributeUsage(AttributeTargets.Parameter, Inherited = false)]
    public sealed class CallerMemberNameAttribute : Attribute
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CallerMemberNameAttribute"/> class.
        /// </summary>
        [TargetedPatchingOptOut("Performance critical to inline this type of method across NGen image boundaries")]
        public CallerMemberNameAttribute()
        {
        }
    }
}

#endif