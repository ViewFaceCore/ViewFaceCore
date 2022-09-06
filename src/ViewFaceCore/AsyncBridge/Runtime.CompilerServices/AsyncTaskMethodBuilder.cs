#if !NET45_OR_GREATER // net40 »ò¸üµÍ°æ±¾

using System.Diagnostics;
using System.Security;
using System.Threading.Tasks;

namespace System.Runtime.CompilerServices
{
    /// <summary>
    /// Provides a builder for asynchronous methods that return <see cref="T:System.Threading.Tasks.Task"/>.
    ///             This type is intended for compiler use only.
    ///
    /// </summary>
    ///
    /// <remarks>
    /// AsyncTaskMethodBuilder is a value type, and thus it is copied by value.
    ///             Prior to being copied, one of its Task, SetResult, or SetException members must be accessed,
    ///             or else the copies may end up building distinct Task instances.
    ///
    /// </remarks>
    public struct AsyncTaskMethodBuilder : IAsyncMethodBuilder
    {
        /// <summary>
        /// A cached VoidTaskResult task used for builders that complete synchronously.
        /// </summary>
        private static readonly TaskCompletionSource<VoidTaskResult> CachedCompleted = AsyncTaskMethodBuilder<VoidTaskResult>.DefaultResultTask;

        /// <summary>
        /// The generic builder object to which this non-generic instance delegates.
        /// </summary>
#pragma warning disable 649
        private AsyncTaskMethodBuilder<VoidTaskResult> builder;
#pragma warning restore 649

        /// <summary>
        /// Gets the <see cref="T:System.Threading.Tasks.Task"/> for this builder.
        /// </summary>
        ///
        /// <returns>
        /// The <see cref="T:System.Threading.Tasks.Task"/> representing the builder's asynchronous operation.
        /// </returns>
        /// <exception cref="T:System.InvalidOperationException">The builder is not initialized.</exception>
        public Task Task
        {
            get
            {
                return builder.Task;
            }
        }

        /// <summary>
        /// Gets an object that may be used to uniquely identify this builder to the debugger.
        ///
        /// </summary>
        ///
        /// <remarks>
        /// This property lazily instantiates the ID in a non-thread-safe manner.
        ///             It must only be used by the debugger, and only in a single-threaded manner
        ///             when no other threads are in the middle of accessing this property or this.Task.
        ///
        /// </remarks>
// ReSharper disable UnusedMember.Local
        private object ObjectIdForDebugger
        // ReSharper restore UnusedMember.Local
        {
            get
            {
                return Task;
            }
        }

        static AsyncTaskMethodBuilder()
        {
        }

        /// <summary>
        /// Initializes a new <see cref="T:System.Runtime.CompilerServices.AsyncTaskMethodBuilder"/>.
        /// </summary>
        ///
        /// <returns>
        /// The initialized <see cref="T:System.Runtime.CompilerServices.AsyncTaskMethodBuilder"/>.
        /// </returns>
        public static AsyncTaskMethodBuilder Create()
        {
            return new AsyncTaskMethodBuilder();
        }

        /// <summary>
        /// Initiates the builder's execution with the associated state machine.
        /// </summary>
        /// <typeparam name="TStateMachine">Specifies the type of the state machine.</typeparam><param name="stateMachine">The state machine instance, passed by reference.</param>
        [DebuggerStepThrough]
        public void Start<TStateMachine>(ref TStateMachine stateMachine) where TStateMachine : IAsyncStateMachine
        {
            builder.Start(ref stateMachine);
        }

        /// <summary>
        /// Associates the builder with the state machine it represents.
        /// </summary>
        /// <param name="stateMachine">The heap-allocated state machine object.</param><exception cref="T:System.ArgumentNullException">The <paramref name="stateMachine"/> argument was null (Nothing in Visual Basic).</exception><exception cref="T:System.InvalidOperationException">The builder is incorrectly initialized.</exception>
        public void SetStateMachine(IAsyncStateMachine stateMachine)
        {
            builder.SetStateMachine(stateMachine);
        }

        void IAsyncMethodBuilder.PreBoxInitialization()
        {
#pragma warning disable 168
            var task = Task;
#pragma warning restore 168
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
            builder.AwaitOnCompleted(ref awaiter, ref stateMachine);
        }

        /// <summary>
        /// Schedules the specified state machine to be pushed forward when the specified awaiter completes.
        ///
        /// </summary>
        /// <typeparam name="TAwaiter">Specifies the type of the awaiter.</typeparam><typeparam name="TStateMachine">Specifies the type of the state machine.</typeparam><param name="awaiter">The awaiter.</param><param name="stateMachine">The state machine.</param>
        public void AwaitUnsafeOnCompleted<TAwaiter, TStateMachine>(ref TAwaiter awaiter, ref TStateMachine stateMachine)
            where TAwaiter : ICriticalNotifyCompletion
            where TStateMachine : IAsyncStateMachine
        {
            builder.AwaitUnsafeOnCompleted(ref awaiter, ref stateMachine);
        }

        /// <summary>
        /// Completes the <see cref="T:System.Threading.Tasks.Task"/> in the
        ///             <see cref="T:System.Threading.Tasks.TaskStatus">RanToCompletion</see> state.
        ///
        /// </summary>
        /// <exception cref="T:System.InvalidOperationException">The builder is not initialized.</exception><exception cref="T:System.InvalidOperationException">The task has already completed.</exception>
        public void SetResult()
        {
            builder.SetResult(CachedCompleted);
        }

        /// <summary>
        /// Completes the <see cref="T:System.Threading.Tasks.Task"/> in the
        ///             <see cref="T:System.Threading.Tasks.TaskStatus">Faulted</see> state with the specified exception.
        ///
        /// </summary>
        /// <param name="exception">The <see cref="T:System.Exception"/> to use to fault the task.</param><exception cref="T:System.ArgumentNullException">The <paramref name="exception"/> argument is null (Nothing in Visual Basic).</exception><exception cref="T:System.InvalidOperationException">The builder is not initialized.</exception><exception cref="T:System.InvalidOperationException">The task has already completed.</exception>
        public void SetException(Exception exception)
        {
            builder.SetException(exception);
        }

        /// <summary>
        /// Called by the debugger to request notification when the first wait operation
        ///             (await, Wait, Result, etc.) on this builder's task completes.
        ///
        /// </summary>
        /// <param name="enabled">true to enable notification; false to disable a previously set notification.
        ///             </param>
        internal void SetNotificationForWaitCompletion(bool enabled)
        {
            builder.SetNotificationForWaitCompletion(enabled);
        }
    }

    /// <summary>
    /// Provides a builder for asynchronous methods that return <see cref="T:System.Threading.Tasks.Task`1"/>.
    ///             This type is intended for compiler use only.
    ///
    /// </summary>
    ///
    /// <remarks>
    /// AsyncTaskMethodBuilder{TResult} is a value type, and thus it is copied by value.
    ///             Prior to being copied, one of its Task, SetResult, or SetException members must be accessed,
    ///             or else the copies may end up building distinct Task instances.
    ///
    /// </remarks>
    public struct AsyncTaskMethodBuilder<TResult> : IAsyncMethodBuilder
    {
        /// <summary>
        /// A cached task for default(TResult).
        /// </summary>
        internal static readonly TaskCompletionSource<TResult> DefaultResultTask = AsyncMethodTaskCache<TResult>.CreateCompleted(default(TResult));
        /// <summary>
        /// State related to the IAsyncStateMachine.
        /// </summary>
#pragma warning disable 649
        private AsyncMethodBuilderCore coreState;
#pragma warning restore 649
        /// <summary>
        /// The lazily-initialized task completion source.
        /// </summary>
        private TaskCompletionSource<TResult> task;

        /// <summary>
        /// Gets the lazily-initialized TaskCompletionSource.
        /// </summary>
        internal TaskCompletionSource<TResult> CompletionSource
        {
            get
            {
                var completionSource = task;
                if (completionSource == null)
                    task = completionSource = new TaskCompletionSource<TResult>();
                return completionSource;
            }
        }

        /// <summary>
        /// Gets the <see cref="T:System.Threading.Tasks.Task`1"/> for this builder.
        /// </summary>
        ///
        /// <returns>
        /// The <see cref="T:System.Threading.Tasks.Task`1"/> representing the builder's asynchronous operation.
        /// </returns>
        public Task<TResult> Task
        {
            get
            {
                return CompletionSource.Task;
            }
        }

        /// <summary>
        /// Gets an object that may be used to uniquely identify this builder to the debugger.
        ///
        /// </summary>
        ///
        /// <remarks>
        /// This property lazily instantiates the ID in a non-thread-safe manner.
        ///             It must only be used by the debugger, and only in a single-threaded manner
        ///             when no other threads are in the middle of accessing this property or this.Task.
        ///
        /// </remarks>
// ReSharper disable UnusedMember.Local
        private object ObjectIdForDebugger
        // ReSharper restore UnusedMember.Local
        {
            get
            {
                return Task;
            }
        }

        /// <summary>
        /// Temporary support for disabling crashing if tasks go unobserved.
        /// </summary>
        static AsyncTaskMethodBuilder()
        {
            try
            {
                AsyncVoidMethodBuilder.PreventUnobservedTaskExceptions();
            }
            catch
            {
            }
        }

        /// <summary>
        /// Initializes a new <see cref="T:System.Runtime.CompilerServices.AsyncTaskMethodBuilder"/>.
        /// </summary>
        ///
        /// <returns>
        /// The initialized <see cref="T:System.Runtime.CompilerServices.AsyncTaskMethodBuilder"/>.
        /// </returns>
        public static AsyncTaskMethodBuilder<TResult> Create()
        {
            return new AsyncTaskMethodBuilder<TResult>();
        }

        /// <summary>
        /// Initiates the builder's execution with the associated state machine.
        /// </summary>
        /// <typeparam name="TStateMachine">Specifies the type of the state machine.</typeparam><param name="stateMachine">The state machine instance, passed by reference.</param>
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
#pragma warning disable 168
            var task = Task;
#pragma warning restore 168
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
        /// Completes the <see cref="T:System.Threading.Tasks.Task`1"/> in the
        ///             <see cref="T:System.Threading.Tasks.TaskStatus">RanToCompletion</see> state with the specified result.
        ///
        /// </summary>
        /// <param name="result">The result to use to complete the task.</param><exception cref="T:System.InvalidOperationException">The task has already completed.</exception>
        public void SetResult(TResult result)
        {
            var completionSource = task;
            if (completionSource == null)
                task = GetTaskForResult(result);
            else if (!completionSource.TrySetResult(result))
                throw new InvalidOperationException("The Task was already completed.");
        }

        /// <summary>
        /// Completes the builder by using either the supplied completed task, or by completing
        ///             the builder's previously accessed task using default(TResult).
        ///
        /// </summary>
        /// <param name="completedTask">A task already completed with the value default(TResult).</param><exception cref="T:System.InvalidOperationException">The task has already completed.</exception>
        internal void SetResult(TaskCompletionSource<TResult> completedTask)
        {
            if (task == null)
                task = completedTask;
            else
                SetResult(default(TResult));
        }

        /// <summary>
        /// Completes the <see cref="T:System.Threading.Tasks.Task`1"/> in the
        ///             <see cref="T:System.Threading.Tasks.TaskStatus">Faulted</see> state with the specified exception.
        ///
        /// </summary>
        /// <param name="exception">The <see cref="T:System.Exception"/> to use to fault the task.</param><exception cref="T:System.ArgumentNullException">The <paramref name="exception"/> argument is null (Nothing in Visual Basic).</exception><exception cref="T:System.InvalidOperationException">The task has already completed.</exception>
        public void SetException(Exception exception)
        {
            if (exception == null)
                throw new ArgumentNullException("exception");
            var completionSource = CompletionSource;
            var setException = (exception is OperationCanceledException ? completionSource.TrySetCanceled() : completionSource.TrySetException(exception));
            if (!setException)
                throw new InvalidOperationException("The Task was already completed.");
        }

        /// <summary>
        /// Called by the debugger to request notification when the first wait operation
        ///             (await, Wait, Result, etc.) on this builder's task completes.
        ///
        /// </summary>
        /// <param name="enabled">true to enable notification; false to disable a previously set notification.
        ///             </param>
        /// <remarks>
        /// This should only be invoked from within an asynchronous method,
        ///             and only by the debugger.
        ///
        /// </remarks>
        internal void SetNotificationForWaitCompletion(bool enabled)
        {
        }

        /// <summary>
        /// Gets a task for the specified result.  This will either
        ///             be a cached or new task, never null.
        ///
        /// </summary>
        /// <param name="result">The result for which we need a task.</param>
        /// <returns>
        /// The completed task containing the result.
        /// </returns>
        private TaskCompletionSource<TResult> GetTaskForResult(TResult result)
        {
            var asyncMethodTaskCache = AsyncMethodTaskCache<TResult>.Singleton;
            if (asyncMethodTaskCache == null)
                return AsyncMethodTaskCache<TResult>.CreateCompleted(result);
            return asyncMethodTaskCache.FromResult(result);
        }
    }
}

#endif