#if !NET45_OR_GREATER // net40 »ò¸üµÍ°æ±¾

using System.Diagnostics;
using System.Security;
using System.Threading;

namespace System.Runtime.CompilerServices
{
    /// <summary>
    /// Holds state related to the builder's IAsyncStateMachine.
    /// </summary>
    ///
    /// <remarks>
    /// This is a mutable struct.  Be very delicate with it.
    /// </remarks>
    internal struct AsyncMethodBuilderCore
    {
        /// <summary>
        /// A reference to the heap-allocated state machine object associated with this builder.
        /// </summary>
        private IAsyncStateMachine stateMachine;

        /// <summary>
        /// Initiates the builder's execution with the associated state machine.
        /// </summary>
        /// <typeparam name="TStateMachine">Specifies the type of the state machine.</typeparam><param name="stateMachine">The state machine instance, passed by reference.</param><exception cref="T:System.ArgumentNullException">The <paramref name="stateMachine"/> argument is null (Nothing in Visual Basic).</exception>
        [SecuritySafeCritical]
        [DebuggerStepThrough]
        internal void Start<TStateMachine>(ref TStateMachine stateMachine) where TStateMachine : IAsyncStateMachine
        {
            // ReSharper disable RedundantCast
            if ((object)stateMachine == null)
                // ReSharper restore RedundantCast
                throw new ArgumentNullException("stateMachine");
            stateMachine.MoveNext();
        }

        /// <summary>
        /// Associates the builder with the state machine it represents.
        /// </summary>
        /// <param name="stateMachine">The heap-allocated state machine object.</param><exception cref="T:System.ArgumentNullException">The <paramref name="stateMachine"/> argument was null (Nothing in Visual Basic).</exception><exception cref="T:System.InvalidOperationException">The builder is incorrectly initialized.</exception>
        public void SetStateMachine(IAsyncStateMachine stateMachine)
        {
            if (stateMachine == null)
                throw new ArgumentNullException("stateMachine");
            if (this.stateMachine != null)
                throw new InvalidOperationException("The builder was not properly initialized.");
            this.stateMachine = stateMachine;
        }

        /// <summary>
        /// Gets the Action to use with an awaiter's OnCompleted or UnsafeOnCompleted method.
        ///             On first invocation, the supplied state machine will be boxed.
        ///
        /// </summary>
        /// <typeparam name="TMethodBuilder">Specifies the type of the method builder used.</typeparam><typeparam name="TStateMachine">Specifies the type of the state machine used.</typeparam><param name="builder">The builder.</param><param name="stateMachine">The state machine.</param>
        /// <returns>
        /// An Action to provide to the awaiter.
        /// </returns>
        [SecuritySafeCritical]
        internal Action GetCompletionAction<TMethodBuilder, TStateMachine>(ref TMethodBuilder builder, ref TStateMachine stateMachine)
            where TMethodBuilder : IAsyncMethodBuilder
            where TStateMachine : IAsyncStateMachine
        {
            var moveNextRunner = new MoveNextRunner(ExecutionContext.Capture());
            Action action = moveNextRunner.Run;
            if (this.stateMachine == null)
            {
                builder.PreBoxInitialization();
                this.stateMachine = stateMachine;
                this.stateMachine.SetStateMachine(this.stateMachine);
            }
            moveNextRunner.StateMachine = this.stateMachine;
            return action;
        }

        /// <summary>
        /// Throws the exception on the ThreadPool.
        /// </summary>
        /// <param name="exception">The exception to propagate.</param><param name="targetContext">The target context on which to propagate the exception.  Null to use the ThreadPool.</param>
        internal static void ThrowAsync(Exception exception, SynchronizationContext targetContext)
        {
            if (targetContext != null)
            {
                try
                {
                    targetContext.Post(state =>
                    {
                        throw TaskAwaiter.PrepareExceptionForRethrow((Exception)state);
                    }, exception);
                    return;
                }
                catch (Exception ex)
                {
                    exception = new AggregateException(new[] { exception, ex });
                }
            }
            ThreadPool.QueueUserWorkItem(state =>
            {
                throw TaskAwaiter.PrepareExceptionForRethrow((Exception)state);
            }, exception);
        }

        /// <summary>
        /// Provides the ability to invoke a state machine's MoveNext method under a supplied ExecutionContext.
        /// </summary>
        private sealed class MoveNextRunner
        {
            /// <summary>
            /// The context with which to run MoveNext.
            /// </summary>
            private readonly ExecutionContext context;
            /// <summary>
            /// The state machine whose MoveNext method should be invoked.
            /// </summary>
            internal IAsyncStateMachine StateMachine;
            /// <summary>
            /// Cached delegate used with ExecutionContext.Run.
            /// </summary>
            [SecurityCritical]
            private static ContextCallback invokeMoveNext;

            /// <summary>
            /// Initializes the runner.
            /// </summary>
            /// <param name="context">The context with which to run MoveNext.</param>
            [SecurityCritical]
            internal MoveNextRunner(ExecutionContext context)
            {
                this.context = context;
            }

            /// <summary>
            /// Invokes MoveNext under the provided context.
            /// </summary>
            [SecuritySafeCritical]
            internal void Run()
            {
                if (context == null)
                {
                    StateMachine.MoveNext();
                    return;
                }

                try
                {
                    var callback = invokeMoveNext;
                    if (callback == null)
                        invokeMoveNext = callback = InvokeMoveNext;
                    ExecutionContext.Run(context, callback, StateMachine);
                }
                finally
                {
#if !NET35
                    context.Dispose();
#endif
                }
            }

            /// <summary>
            /// Invokes the MoveNext method on the supplied IAsyncStateMachine.
            /// </summary>
            /// <param name="stateMachine">The IAsyncStateMachine machine instance.</param>
            [SecurityCritical]
            private static void InvokeMoveNext(object stateMachine)
            {
                ((IAsyncStateMachine)stateMachine).MoveNext();
            }
        }
    }
}

#endif