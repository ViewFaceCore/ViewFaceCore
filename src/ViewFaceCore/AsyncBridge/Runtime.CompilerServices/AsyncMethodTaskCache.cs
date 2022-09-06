#if !NET45_OR_GREATER // net40 »ò¸üµÍ°æ±¾

using System.Threading.Tasks;

namespace System.Runtime.CompilerServices
{
    /// <summary>
    /// Provides a base class used to cache tasks of a specific return type.
    /// </summary>
    /// <typeparam name="TResult">Specifies the type of results the cached tasks return.</typeparam>
    internal class AsyncMethodTaskCache<TResult>
    {
        /// <summary>
        /// A singleton cache for this result type.
        ///             This may be null if there are no cached tasks for this TResult.
        ///
        /// </summary>
        internal static readonly AsyncMethodTaskCache<TResult> Singleton = CreateCache();

        static AsyncMethodTaskCache()
        {
        }

        /// <summary>
        /// Creates a non-disposable task.
        /// </summary>
        /// <param name="result">The result for the task.</param>
        /// <returns>
        /// The cacheable task.
        /// </returns>
        internal static TaskCompletionSource<TResult> CreateCompleted(TResult result)
        {
            var completionSource = new TaskCompletionSource<TResult>();
            completionSource.TrySetResult(result);
            return completionSource;
        }

        /// <summary>
        /// Creates a cache.
        /// </summary>
        ///
        /// <returns>
        /// A task cache for this result type.
        /// </returns>
        private static AsyncMethodTaskCache<TResult> CreateCache()
        {
            var type = typeof(TResult);
            if (type == typeof(bool))
                return (AsyncMethodTaskCache<TResult>)(object)new AsyncMethodBooleanTaskCache();
            if (type == typeof(int))
                return (AsyncMethodTaskCache<TResult>)(object)new AsyncMethodInt32TaskCache();

            return null;
        }

        /// <summary>
        /// Gets a cached task if one exists.
        /// </summary>
        /// <param name="result">The result for which we want a cached task.</param>
        /// <returns>
        /// A cached task if one exists; otherwise, null.
        /// </returns>
        internal virtual TaskCompletionSource<TResult> FromResult(TResult result)
        {
            return CreateCompleted(result);
        }

        /// <summary>
        /// Provides a cache for Boolean tasks.
        /// </summary>
        private sealed class AsyncMethodBooleanTaskCache : AsyncMethodTaskCache<bool>
        {
            /// <summary>
            /// A true task.
            /// </summary>
            private static readonly TaskCompletionSource<bool> True = CreateCompleted(true);
            /// <summary>
            /// A false task.
            /// </summary>
            private static readonly TaskCompletionSource<bool> False = CreateCompleted(false);

            /// <summary>
            /// Gets a cached task for the Boolean result.
            /// </summary>
            /// <param name="result">true or false</param>
            /// <returns>
            /// A cached task for the Boolean result.
            /// </returns>
            internal override TaskCompletionSource<bool> FromResult(bool result)
            {
                return result ? True : False;
            }
        }

        /// <summary>
        /// Provides a cache for zero Int32 tasks.
        /// </summary>
        private sealed class AsyncMethodInt32TaskCache : AsyncMethodTaskCache<int>
        {
            /// <summary>
            /// The cache of Task{Int32}.
            /// </summary>
            private static readonly TaskCompletionSource<int>[] Int32Tasks = CreateInt32Tasks();
            /// <summary>
            /// The minimum value, inclusive, for which we want a cached task.
            /// </summary>
            private const int InclusiveInt32Min = -1;
            /// <summary>
            /// The maximum value, exclusive, for which we want a cached task.
            /// </summary>
            private const int ExclusiveInt32Max = 9;

            static AsyncMethodInt32TaskCache()
            {
            }

            /// <summary>
            /// Creates an array of cached tasks for the values in the range [INCLUSIVE_MIN,EXCLUSIVE_MAX).
            /// </summary>
            private static TaskCompletionSource<int>[] CreateInt32Tasks()
            {
                var completionSourceArray = new TaskCompletionSource<int>[10];
                for (var index = 0; index < completionSourceArray.Length; ++index)
                    completionSourceArray[index] = CreateCompleted(index - 1);
                return completionSourceArray;
            }

            /// <summary>
            /// Gets a cached task for the zero Int32 result.
            /// </summary>
            /// <param name="result">The integer value</param>
            /// <returns>
            /// A cached task for the Int32 result or null if not cached.
            /// </returns>
            internal override TaskCompletionSource<int> FromResult(int result)
            {
                if (result < InclusiveInt32Min || result >= ExclusiveInt32Max)
                    return CreateCompleted(result);

                return Int32Tasks[result - -1];
            }
        }
    }
}

#endif