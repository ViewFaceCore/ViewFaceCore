#if !NET45_OR_GREATER // net40 »ò¸üµÍ°æ±¾

#if NET35 || PORTABLE

namespace System.Runtime
{
    /// <summary>
    /// Indicates that the .NET Framework class library method to which this attribute is applied is unlikely to be affected by servicing releases,
    /// and therefore is eligible to be inlined across Native Image Generator (NGen) images.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Constructor, AllowMultiple = false, Inherited = false)]
    internal sealed class TargetedPatchingOptOutAttribute : Attribute
    {
        /// <summary>
        /// Infrastructure. Initializes a new instance of the <b>TargetedPatchingOptOutAttribute</b> class.
        /// </summary>
        /// <param name="reason">The reason why the method to which the <see cref="TargetedPatchingOptOutAttribute"/> attribute is applied is
        /// considered to be eligible for inlining across Native Image Generator (NGen) images.</param>
        public TargetedPatchingOptOutAttribute(string reason)
        {
            Reason = reason;
        }

        /// <summary>
        /// Infrastructure. Gets the reason why the method to which this attribute is applied is considered to be eligible for inlining across
        /// Native Image Generator (NGen) images.
        /// </summary>
        public string Reason { get; }
    }
}

#endif

#endif