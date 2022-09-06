#if !NET45_OR_GREATER // net40 »ò¸üµÍ°æ±¾

using System.Diagnostics;
using System.Security;
using System.Threading;
using System.Threading.Tasks;

namespace System.Runtime.CompilerServices
{
    /// <summary>
    /// Provides a builder for asynchronous methods that return void.
    ///             This type is intended for compiler use only.
    ///
    /// </summary>
    public struct AsyncVoidMethodBuilder : IAsyncMethodBuilder
    {
        /// <summary>
        /// The synchronization context associated with this operation.
        /// </summary>
        private readonly SynchronizationContext synchronizationContext;
        /// <summary>
        /// State related to the IAsyncStateMachine.
        /// </summary>
        private AsyncMethodBuilderCore coreState;
        /// <summary>
        /// An object used by the debugger to uniquely identify this builder.  Lazily initialized.
        /// </summary>
        private object objectIdForDebugger;
        /// <summary>
        /// Non-zero if PreventUnobservedTaskExceptions has already been invoked.
        /// </summary>
        private static int preventUnobservedTaskExceptionsInvoked;

        /// <summary>
        /// Gets an object that may be used to uniquely identify this builder to the debugger.
        ///
        /// </summary>
        ///
        /// <remarks>
        /// This property lazily instantiates the ID in a non-thread-safe manner.
        ///             It must only be used by the debugger and only in a single-threaded manner.
        ///
        /// </remarks>
// ReSharper disable UnusedMember.Local
        private object ObjectIdForDebugger
        // ReSharper restore UnusedMember.Local
        {
            get
            {
                return objectIdForDebugger ?? (objectIdForDebugger = new object());
            }
        }

        /// <summary>
        /// Temporary support for disabling crashing if tasks go unobserved.
        /// </summary>
        static AsyncVoidMethodBuilder()
        {
            try
            {
                PreventUnobservedTaskExceptions();
            }
            catch
            {
            }
        }

        /// <summary>
        /// Initializes the <see cref="T:System.Runtime.CompilerServices.AsyncVoidMethodBuilder"/>.
        /// </summary>
        /// <param name="synchronizationContext">The synchronizationContext associated with this operation. This may be null.</param>
        private AsyncVoidMethodBuilder(SynchronizationContext synchronizationContext)
        {
            this.synchronizationContext = synchronizationContext;
            if (synchronizationContext != null)
                synchronizationContext.OperationStarted();
            coreState = new AsyncMethodBuilderCore();
            objectIdForDebugger = null;
        }

        /// <summary>
        /// Registers with UnobservedTaskException to suppress exception crashing.
        /// </summary>
        internal static void PreventUnobservedTaskExceptions()
        {
            if (Interlocked.CompareExchange(ref preventUnobservedTaskExceptionsInvoked, 1, 0) != 0)
                return;
            TaskScheduler.UnobservedTaskException += (EventHandler<UnobservedTaskExceptionEventArgs>)((s, e) => e.SetObserved());
        }

        /// <summary>
        /// Initializes a new <see cref="T:System.Runtime.CompilerServices.AsyncVoidMethodBuilder"/>.
        /// </summary>
        ///
        /// <returns>
        /// The initialized <see cref="T:System.Runtime.CompilerServices.AsyncVoidMethodBuilder"/>.
        /// </returns>
        public static AsyncVoidMethodBuilder Create()
        {
            return new AsyncVoidMethodBuilder(SynchronizationContext.Current);
        }

        /// <summary>
        /// Initiates the builder's execution with the associated state machine.
        /// </summary>
        /// <typeparam name="TStateMachine">Specifies the type of the state machine.</typeparam><param name="stateMachine">The state machine instance, passed by reference.</param><exception cref="T:System.ArgumentNullException">The <paramref name="stateMachine"/> argument was null (Nothing in Visual Basic).</exception>
        [DebuggerStepThrough]
        public void Start<TStateMachine>(ref TStateMachine stateMachine) where TStateMachine : IAsyncStateMachine
        {
            coreState.Start(ref stateMachine);
        }

        /// <summary>
        /// Associates the builder with the state machine it represents.
        /// </summary>
        /// <param name="stateMachine">The heap-allocated state machine object.</param><exception cref="T:System.ArgumentNullException">The <paramref name="stateMachine"/> argument was null (Nothing in Visual Basic).</exception><exception cref="T:System.InvalidOperationException">The builder is incorrectly initialized.</exception>
        public void SetStateMachine(IAsyncStateMachine stateMachine)
        {
            coreState.SetStateMachine(stateMachine);
        }

        void IAsyncMethodBuilder.PreBoxInitialization()
        {
        }

        /// <summary>
        /// Schedules the specified state machine to be pushed forward when the specified awaiter completes.
        ///
        /// </summary>
        /// <typeparam name="TAwaiter">Specifies the type of the awaiter.</typeparam><typeparam name="TStateMachine">Specifies the type of the state machine.</typeparam><param name="awaiter">The awaiter.</param><param name="stateMachine">The state machine.</param>
        public void AwaitOnCompleted<TAwaiter, TStateMachine>(ref TAwaiter awaiter, ref TStateMachine stateMachine)
            where TAwaiter : INotifyCompletion
            where TStateMachine : IAsyncStateMachine
        {
            try
            {
                var completionAction = coreState.GetCompletionAction(ref this, ref stateMachine);
                awaiter.OnCompleted(completionAction);
            }
            catch (Exception ex)
            {
                AsyncMethodBuilderCore.ThrowAsync(ex, null);
            }
        }

        /// <summary>
        /// Schedules the specified state machine to be pushed forward when the specified awaiter completes.
        ///
        /// </summary>
        /// <typeparam name="TAwaiter">Specifies the type of the awaiter.</typeparam><typeparam name="TStateMachine">Specifies the type of the state machine.</typeparam><param name="awaiter">The awaiter.</param><param name="stateMachine">The state machine.</param>
        [SecuritySafeCritical]
        public void AwaitUnsafeOnCompleted<TAwaiter, TStateMachine>(ref TAwaiter awaiter, ref TStateMachine stateMachine)
            where TAwaiter : ICriticalNotifyCompletion
            where TStateMachine : IAsyncStateMachine
        {
            try
            {
                var completionAction = coreState.GetCompletionAction(ref this, ref stateMachine);
                awaiter.UnsafeOnCompleted(completionAction);
            }
            catch (Exception ex)
            {
                AsyncMethodBuilderCore.ThrowAsync(ex, null);
            }
        }

        /// <summary>
        /// Completes the method builder successfully.
        /// </summary>
        public void SetResult()
        {
            if (synchronizationContext == null)
                return;
            NotifySynchronizationContextOfCompletion();
        }

        /// <summary>
        /// Faults the method builder with an exception.
        /// </summary>
        /// <param name="exception">The exception that is the cause of this fault.</param><exception cref="T:System.ArgumentNullException">The <paramref name="exception"/> argument is null (Nothing in Visual Basic).</exception><exception cref="T:System.InvalidOperationException">The builder is not initialized.</exception>
        public void SetException(Exception exception)
        {
            if (exception == null)
                throw new ArgumentNullException("exception");
            if (synchronizationContext != null)
            {
                try
                {
                    AsyncMethodBuilderCore.ThrowAsync(exception, synchronizationContext);
                }
                finally
                {
                    NotifySynchronizationContextOfCompletion();
                }
            }
            else
                AsyncMethodBuilderCore.ThrowAsync(exception, null);
        }

        /// <summary>
        /// Notifies the current synchronization context that the operation completed.
        /// </summary>
        private void NotifySynchronizationContextOfCompletion()
        {
            try
            {
                synchronizationContext.OperationCompleted();
            }
            catch (Exception ex)
            {
                AsyncMethodBuilderCore.ThrowAsync(ex, null);
            }
        }
    }
}

#endif