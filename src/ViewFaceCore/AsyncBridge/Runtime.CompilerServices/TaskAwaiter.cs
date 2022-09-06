#if !NET45_OR_GREATER // net40 »ò¸üµÍ°æ±¾

using System.Reflection;
using System.Security;
using System.Threading;
using System.Threading.Tasks;

namespace System.Runtime.CompilerServices
{
    /// <summary>
    ///   Provides an awaiter for awaiting a <see cref="Task" /> .
    /// </summary>
    /// <remarks>
    ///   This type is intended for compiler use only.
    /// </remarks>
    public struct TaskAwaiter : ICriticalNotifyCompletion
    {
        /// <summary>
        ///   A MethodInfo for the Exception.PrepForRemoting method.
        /// </summary>
        private static readonly MethodInfo PrepForRemoting = GetPrepForRemotingMethodInfo();

        /// <summary>
        ///   An empty array to use with MethodInfo.Invoke.
        /// </summary>
        private static readonly object[] EmptyParams = new object[0];

        /// <summary>
        ///   The default value to use for continueOnCapturedContext.
        /// </summary>
        internal const bool ContinueOnCapturedContextDefault = true;

        /// <summary>
        ///   Error message for GetAwaiter.
        /// </summary>
        private const string InvalidOperationExceptionTaskNotCompleted = "The task has not yet completed.";

        /// <summary>
        ///   The task being awaited.
        /// </summary>
        private readonly Task task;

        /// <summary>
        ///   Gets whether the task being awaited is completed.
        /// </summary>
        /// <remarks>
        ///   This property is intended for compiler user rather than use directly in code.
        /// </remarks>
        /// <exception cref="NullReferenceException">The awaiter was not properly initialized.</exception>
        public bool IsCompleted
        {
            get { return task.IsCompleted; }
        }

        /// <summary>
        ///   Whether the current thread is appropriate for inlining the await continuation.
        /// </summary>
        private static bool IsValidLocationForInlining
        {
            get
            {
                var current = SynchronizationContext.Current;
                if (current != null && current.GetType() != typeof(SynchronizationContext))
                    return false;
                return TaskScheduler.Current == TaskScheduler.Default;
            }
        }

        static TaskAwaiter()
        {
        }

        /// <summary>
        ///   Initializes the <see cref="TaskAwaiter" /> .
        /// </summary>
        /// <param name="task"> The <see cref="Task" /> to be awaited. </param>
        internal TaskAwaiter(Task task)
        {
            this.task = task;
        }

        /// <summary>
        ///   Schedules the continuation onto the <see cref="Task" /> associated with this <see cref="TaskAwaiter" /> .
        /// </summary>
        /// <param name="continuation"> The action to invoke when the await operation completes. </param>
        /// <exception cref="ArgumentNullException">The
        ///   <paramref name="continuation" />
        ///   argument is null (Nothing in Visual Basic).</exception>
        /// <exception cref="InvalidOperationException">The awaiter was not properly initialized.</exception>
        /// <remarks>
        ///   This method is intended for compiler user rather than use directly in code.
        /// </remarks>
        public void OnCompleted(Action continuation)
        {
            OnCompletedInternal(task, continuation, true);
        }

        /// <summary>
        ///   Schedules the continuation onto the <see cref="Task" /> associated with this <see cref="TaskAwaiter" /> .
        /// </summary>
        /// <param name="continuation"> The action to invoke when the await operation completes. </param>
        /// <exception cref="ArgumentNullException">The
        ///   <paramref name="continuation" />
        ///   argument is null (Nothing in Visual Basic).</exception>
        /// <exception cref="InvalidOperationException">The awaiter was not properly initialized.</exception>
        /// <remarks>
        ///   This method is intended for compiler user rather than use directly in code.
        /// </remarks>
        [SecurityCritical]
        public void UnsafeOnCompleted(Action continuation)
        {
            OnCompletedInternal(task, continuation, true);
        }

        /// <summary>
        ///   Ends the await on the completed <see cref="Task" /> .
        /// </summary>
        /// <exception cref="NullReferenceException">The awaiter was not properly initialized.</exception>
        /// <exception cref="InvalidOperationException">The task was not yet completed.</exception>
        /// <exception cref="TaskCanceledException">The task was canceled.</exception>
        /// <exception cref="Exception">The task completed in a Faulted state.</exception>
        public void GetResult()
        {
            ValidateEnd(task);
        }

        /// <summary>
        ///   Fast checks for the end of an await operation to determine whether more needs to be done prior to completing the await.
        /// </summary>
        /// <param name="task"> The awaited task. </param>
        internal static void ValidateEnd(Task task)
        {
            if (task.Status == TaskStatus.RanToCompletion)
                return;
            HandleNonSuccess(task);
        }

        /// <summary>
        ///   Handles validations on tasks that aren't successfully completed.
        /// </summary>
        /// <param name="task"> The awaited task. </param>
        private static void HandleNonSuccess(Task task)
        {
            if (!task.IsCompleted)
            {
                try
                {
                    task.Wait();
                }
                catch
                {
                }
            }
            if (task.Status == TaskStatus.RanToCompletion)
                return;
            ThrowForNonSuccess(task);
        }

        /// <summary>
        ///   Throws an exception to handle a task that completed in a state other than RanToCompletion.
        /// </summary>
        private static void ThrowForNonSuccess(Task task)
        {
            switch (task.Status)
            {
                case TaskStatus.Canceled:
                    throw new TaskCanceledException(task);
                case TaskStatus.Faulted:
                    throw PrepareExceptionForRethrow(task.Exception.InnerException);
                default:
                    throw new InvalidOperationException("The task has not yet completed.");
            }
        }

        /// <summary>
        ///   Schedules the continuation onto the <see cref="Task" /> associated with this <see cref="TaskAwaiter" /> .
        /// </summary>
        /// <param name="task"> The awaited task. </param>
        /// <param name="continuation"> The action to invoke when the await operation completes. </param>
        /// <param name="continueOnCapturedContext"> Whether to capture and marshal back to the current context. </param>
        /// <exception cref="ArgumentNullException">The
        ///   <paramref name="continuation" />
        ///   argument is null (Nothing in Visual Basic).</exception>
        /// <exception cref="NullReferenceException">The awaiter was not properly initialized.</exception>
        /// <remarks>
        ///   This method is intended for compiler user rather than use directly in code.
        /// </remarks>
        internal static void OnCompletedInternal(Task task, Action continuation, bool continueOnCapturedContext)
        {
            if (continuation == null)
                throw new ArgumentNullException("continuation");
            var syncContext = continueOnCapturedContext ? SynchronizationContext.Current : null;
            if (syncContext != null && syncContext.GetType() != typeof(SynchronizationContext))
            {
                task.ContinueWith(result =>
                {
                    try
                    {
                        syncContext.Post(state => ((Action)state)(), continuation);
                    }
                    catch (Exception ex)
                    {
                        AsyncMethodBuilderCore.ThrowAsync(ex, null);
                    }
                }, CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
            }
            else
            {
                var scheduler = continueOnCapturedContext ? TaskScheduler.Current : TaskScheduler.Default;
                if (task.IsCompleted)
                {
                    Task.Factory.StartNew(state => ((Action)state)(), continuation, CancellationToken.None,
                                          TaskCreationOptions.None, scheduler);
                }
                else if (scheduler != TaskScheduler.Default)
                {
                    task.ContinueWith(_ => RunNoException(continuation), CancellationToken.None,
                                      TaskContinuationOptions.ExecuteSynchronously, scheduler);
                }
                else
                {
                    task.ContinueWith(result =>
                    {
                        if (IsValidLocationForInlining)
                        {
                            RunNoException(continuation);
                        }
                        else
                        {
                            Task.Factory.StartNew(state => RunNoException((Action)state), continuation,
                                                  CancellationToken.None, TaskCreationOptions.None,
                                                  TaskScheduler.Default);
                        }
                    }, CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
                }
            }
        }

        /// <summary>
        ///   Invokes the delegate in a try/catch that will propagate the exception asynchronously on the ThreadPool.
        /// </summary>
        /// <param name="continuation" />
        private static void RunNoException(Action continuation)
        {
            try
            {
                continuation();
            }
            catch (Exception ex)
            {
                AsyncMethodBuilderCore.ThrowAsync(ex, null);
            }
        }

        /// <summary>
        ///   Copies the exception's stack trace so its stack trace isn't overwritten.
        /// </summary>
        /// <param name="exc"> The exception to prepare. </param>
        internal static Exception PrepareExceptionForRethrow(Exception exc)
        {
            if (PrepForRemoting != null)
            {
                try
                {
                    PrepForRemoting.Invoke(exc, EmptyParams);
                }
                catch
                {
                }
            }
            return exc;
        }

        /// <summary>
        ///   Gets the MethodInfo for the internal PrepForRemoting method on Exception.
        /// </summary>
        /// <returns> The MethodInfo if it could be retrieved, or else null. </returns>
        private static MethodInfo GetPrepForRemotingMethodInfo()
        {
            try
            {
                return typeof(Exception).GetMethod("PrepForRemoting", BindingFlags.Instance | BindingFlags.NonPublic);
            }
            catch
            {
                return null;
            }
        }
    }

    /// <summary>
    ///   Provides an awaiter for awaiting a <see cref="Task{TResult}" /> .
    /// </summary>
    /// <remarks>
    ///   This type is intended for compiler use only.
    /// </remarks>
    public struct TaskAwaiter<TResult> : ICriticalNotifyCompletion
    {
        /// <summary>
        ///   The task being awaited.
        /// </summary>
        private readonly Task<TResult> task;

        /// <summary>
        ///   Gets whether the task being awaited is completed.
        /// </summary>
        /// <remarks>
        ///   This property is intended for compiler user rather than use directly in code.
        /// </remarks>
        /// <exception cref="NullReferenceException">The awaiter was not properly initialized.</exception>
        public bool IsCompleted
        {
            get { return task.IsCompleted; }
        }

        /// <summary>
        ///   Initializes the <see cref="TaskAwaiter{TResult}" /> .
        /// </summary>
        /// <param name="task"> The <see cref="Task{TResult}" /> to be awaited. </param>
        internal TaskAwaiter(Task<TResult> task)
        {
            this.task = task;
        }

        /// <summary>
        ///   Schedules the continuation onto the <see cref="Task" /> associated with this <see cref="TaskAwaiter" /> .
        /// </summary>
        /// <param name="continuation"> The action to invoke when the await operation completes. </param>
        /// <exception cref="ArgumentNullException">The
        ///   <paramref name="continuation" />
        ///   argument is null (Nothing in Visual Basic).</exception>
        /// <exception cref="NullReferenceException">The awaiter was not properly initialized.</exception>
        /// <remarks>
        ///   This method is intended for compiler user rather than use directly in code.
        /// </remarks>
        public void OnCompleted(Action continuation)
        {
            TaskAwaiter.OnCompletedInternal(task, continuation, true);
        }

        /// <summary>
        ///   Schedules the continuation onto the <see cref="Task" /> associated with this <see
        ///    cref="TaskAwaiter" /> .
        /// </summary>
        /// <param name="continuation"> The action to invoke when the await operation completes. </param>
        /// <exception cref="ArgumentNullException">The
        ///   <paramref name="continuation" />
        ///   argument is null (Nothing in Visual Basic).</exception>
        /// <exception cref="InvalidOperationException">The awaiter was not properly initialized.</exception>
        /// <remarks>
        ///   This method is intended for compiler user rather than use directly in code.
        /// </remarks>
        [SecurityCritical]
        public void UnsafeOnCompleted(Action continuation)
        {
            TaskAwaiter.OnCompletedInternal(task, continuation, true);
        }

        /// <summary>
        ///   Ends the await on the completed <see cref="Task{TResult}" /> .
        /// </summary>
        /// <returns> The result of the completed <see cref="Task{TResult}" /> . </returns>
        /// <exception cref="NullReferenceException">The awaiter was not properly initialized.</exception>
        /// <exception cref="InvalidOperationException">The task was not yet completed.</exception>
        /// <exception cref="TaskCanceledException">The task was canceled.</exception>
        /// <exception cref="Exception">The task completed in a Faulted state.</exception>
        public TResult GetResult()
        {
            TaskAwaiter.ValidateEnd(task);
            return task.Result;
        }
    }
}

#endif