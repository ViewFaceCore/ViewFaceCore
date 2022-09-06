#if !NET45_OR_GREATER // net40 »ò¸üµÍ°æ±¾

using System;
#if NET35
using System.Collections.Concurrent;
#endif
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

// ReSharper disable CheckNamespace

/// <summary>
/// Provides extension methods for threading-related types.
/// </summary>
///
/// <summary>
/// Asynchronous wrappers for .NET Framework operations.
/// </summary>
///
/// <summary>
/// Provides extension methods for threading-related types.
/// </summary>
///
/// <remarks>
/// AsyncCtpThreadingExtensions is a placeholder.
/// </remarks>
public static class AsyncCompatLibExtensions
{
#if NET35
    private static readonly ConcurrentDictionary<WeakCTS, WeakReference> _cancelTimers = new ConcurrentDictionary<WeakCTS, WeakReference>();
#else
    private static readonly ConditionalWeakTable<CancellationTokenSource, Timer> _cancelTimers = new ConditionalWeakTable<CancellationTokenSource, Timer>();
#endif
    private static readonly TimeSpan _timeoutInfinite = new TimeSpan(0, 0, 0, 0, -1);

    /// <summary>
    /// Gets an awaiter used to await this <see cref="T:System.Threading.Tasks.Task"/>.
    /// </summary>
    /// <param name="task">The task to await.</param>
    /// <returns>
    /// An awaiter instance.
    /// </returns>
    public static TaskAwaiter GetAwaiter(this Task task)
    {
        if (task == null)
            throw new ArgumentNullException("task");

        return new TaskAwaiter(task);
    }

    /// <summary>
    /// Gets an awaiter used to await this <see cref="T:System.Threading.Tasks.Task"/>.
    /// </summary>
    /// <typeparam name="TResult">Specifies the type of data returned by the task.</typeparam>
    /// <param name="task">The task to await.</param>
    /// <returns>
    /// An awaiter instance.
    /// </returns>
    public static TaskAwaiter<TResult> GetAwaiter<TResult>(this Task<TResult> task)
    {
        if (task == null)
            throw new ArgumentNullException("task");

        return new TaskAwaiter<TResult>(task);
    }

    /// <summary>
    /// Creates and configures an awaitable object for awaiting the specified task.
    /// </summary>
    /// <param name="task">The task to be awaited.</param>
    /// <param name="continueOnCapturedContext">true to automatic marshal back to the original call site's current SynchronizationContext
    ///             or TaskScheduler; otherwise, false.</param>
    /// <returns>
    /// The instance to be awaited.
    /// </returns>
    public static ConfiguredTaskAwaitable<TResult> ConfigureAwait<TResult>(this Task<TResult> task, bool continueOnCapturedContext)
    {
        if (task == null)
            throw new ArgumentNullException("task");

        return new ConfiguredTaskAwaitable<TResult>(task, continueOnCapturedContext);
    }

    /// <summary>
    /// Creates and configures an awaitable object for awaiting the specified task.
    /// </summary>
    /// <param name="task">The task to be awaited.</param>
    /// <param name="continueOnCapturedContext">true to automatic marshal back to the original call site's current SynchronizationContext
    ///             or TaskScheduler; otherwise, false.</param>
    /// <returns>
    /// The instance to be awaited.
    /// </returns>
    public static ConfiguredTaskAwaitable ConfigureAwait(this Task task, bool continueOnCapturedContext)
    {
        if (task == null)
            throw new ArgumentNullException("task");

        return new ConfiguredTaskAwaitable(task, continueOnCapturedContext);
    }

    /// <summary>
    /// Schedules a Cancel operation on this <see cref="T:System.Threading.CancellationTokenSource"/>.
    /// </summary>
    /// <param name="cancelSource">The <see cref="T:System.Threading.CancellationTokenSource"/> to cancel</param>
    /// <param name="millisecondsDelay">The time span to wait before canceling this <see
    /// cref="T:System.Threading.CancellationTokenSource"/>.
    /// </param>
    /// <exception cref="T:System.ObjectDisposedException">The exception thrown when this <see
    /// cref="T:System.Threading.CancellationTokenSource"/> has been disposed.
    /// </exception>
    /// <exception cref="T:System.ArgumentOutOfRangeException">
    /// The exception thrown when <paramref name="millisecondsDelay"/> is less than -1.
    /// </exception>
    /// <remarks>
    /// <para>
    /// The countdown for the millisecondsDelay starts during this call.  When the millisecondsDelay expires, 
    /// this <see cref="T:System.Threading.CancellationTokenSource"/> is canceled, if it has
    /// not been canceled already.
    /// </para>
    /// <para>
    /// Subsequent calls to CancelAfter will reset the millisecondsDelay for this  
    /// <see cref="T:System.Threading.CancellationTokenSource"/>, if it has not been
    /// canceled already.
    /// </para>
    /// </remarks>
    public static void CancelAfter(this CancellationTokenSource cancelSource, int millisecondsDelay)
    {
        if (millisecondsDelay < Timeout.Infinite)
            throw new ArgumentOutOfRangeException(nameof(millisecondsDelay));

        cancelSource.CancelAfter(new TimeSpan(millisecondsDelay * TimeSpan.TicksPerMillisecond));
    }

    /// <summary>
    /// Schedules a Cancel operation on this <see cref="T:System.Threading.CancellationTokenSource"/>.
    /// </summary>
    /// <param name="cancelSource">The <see cref="T:System.Threading.CancellationTokenSource"/> to cancel</param>
    /// <param name="delay">The time span to wait before canceling this <see
    /// cref="T:System.Threading.CancellationTokenSource"/>.
    /// </param>
    /// <exception cref="T:System.ObjectDisposedException">The exception thrown when this <see
    /// cref="T:System.Threading.CancellationTokenSource"/> has been disposed.
    /// </exception>
    /// <exception cref="T:System.ArgumentOutOfRangeException">
    /// The exception thrown when <paramref name="delay"/> is less than -1 or 
    /// greater than Int32.MaxValue.
    /// </exception>
    /// <remarks>
    /// <para>
    /// The countdown for the delay starts during this call.  When the delay expires, 
    /// this <see cref="T:System.Threading.CancellationTokenSource"/> is canceled, if it has
    /// not been canceled already.
    /// </para>
    /// <para>
    /// Subsequent calls to CancelAfter will reset the delay for this  
    /// <see cref="T:System.Threading.CancellationTokenSource"/>, if it has not been
    /// canceled already.
    /// </para>
    /// </remarks>
    public static void CancelAfter(this CancellationTokenSource cancelSource, TimeSpan delay)
    {
        if (cancelSource == null)
            throw new ArgumentNullException(nameof(cancelSource));

        if (delay < _timeoutInfinite)
            throw new ArgumentOutOfRangeException(nameof(delay));

        if (cancelSource.IsCancellationRequested)
            return;

        Timer myTimer = null;

        // CTS claims it's thread-safe for all methods, so we should be the same.
        // If someone calls CancelAfter concurrently for the same CTS, we won't explode or cause a memory leak
        while (!_cancelTimers.TryGetTimer(cancelSource, out myTimer))
        {
            // An active timer doesn't hold a strong reference to itself, so adding the CTS as state won't hold it alive
            myTimer = new Timer(OnCancelAfterTimer, cancelSource, Timeout.Infinite, Timeout.Infinite);

            // There's a tiny chance we may add a new Timer to a cancelled CTS, if the callback is running concurrently.
            // If so, the extra Timer execution will be a NOP (and probably be GC'ed before it even runs).
            if (_cancelTimers.TryAddTimer(cancelSource, myTimer))
                break;

            // TryAddTimer can only fail if a Timer was created concurrently.
            // Dispose of the new Timer we just created, and loop back to change the new one.
            // Of course, in this case it's indeterminate which delay will be the final one, but that's the risk you take calling this on multiple threads.
            myTimer.Dispose();
        }

        try
        {
            // Either set the duration on the new timer, or reset the duration on an existing timer
            myTimer.Change(delay, _timeoutInfinite);
        }
        catch (ObjectDisposedException)
        {
        }
    }

    private static void OnCancelAfterTimer(object state)
    {
        var cancelSource = (CancellationTokenSource)state;

        if (!_cancelTimers.TryRemoveTimer(cancelSource, out var oldTimer))
            return;

        oldTimer.Dispose();

        try
        {
            cancelSource.Cancel();
        }
        catch (ObjectDisposedException) // If the cancellation token has been disposed of, ignore the exception
        {
        }
    }

#if NET35
    private struct WeakCTS : IEquatable<WeakCTS>
    {
        private readonly int _hashCode;

        internal WeakCTS(CancellationTokenSource cts)
        {
            _hashCode = cts.GetHashCode();
            CTS = new WeakReference(cts);
        }

        public override int GetHashCode()
        {
            return _hashCode;
        }

        public override bool Equals(object obj)
        {
            return obj is WeakCTS Value && Equals(Value);
        }

        public bool Equals(WeakCTS other)
        {
            // In the case our CTS is no longer alive, this will still work when doing dictionary lookups. Null matches null
            return _hashCode == other._hashCode && ReferenceEquals(CTS.Target, other.CTS.Target);
        }

        internal WeakReference CTS { get; }

        public static implicit operator WeakCTS(CancellationTokenSource cts)
        {
            return new WeakCTS(cts);
        }
    }

    private static bool TryAddTimer(this ConcurrentDictionary<WeakCTS, WeakReference> table, CancellationTokenSource cancelSource, Timer timer)
    {
        if (!table.TryAdd(cancelSource, new WeakReference(timer)))
            return false;

        // Since we use weak-references to the Timer, we need to ensure it doesn't get garbage collected until the CTS does.
        // The CancellationToken registration works - we just stick it in the state property, and it'll hold a strong reference for us.
        // There's a chance this can throw ObjectDisposedException if someone disposes of our CTS. We'll just leave the orphan Timer to be GC'ed
        cancelSource.Token.Register((state) => { }, timer);

        // Cleanup any dead CTS->Timer links
        foreach (var key in _cancelTimers.Keys)
        {
            if (!key.CTS.IsAlive)
                _cancelTimers.TryRemove(key, out _);
        }

        return true;
    }

    private static bool TryGetTimer(this ConcurrentDictionary<WeakCTS, WeakReference> table, CancellationTokenSource cancelSource, out Timer timer)
    {
        if (!table.TryGetValue(cancelSource, out var weakReference))
        {
            timer = null;
            return false;
        }

        timer = (Timer)weakReference.Target;

        return timer != null; // Could return false if the CTS was cancelled concurrently and our timer triggered
    }

    private static bool TryRemoveTimer(this ConcurrentDictionary<WeakCTS, WeakReference> table, CancellationTokenSource cancelSource, out Timer timer)
    {
        if (!table.TryRemove(cancelSource, out var weakReference))
        {
            timer = null;
            return false;
        }

        timer = (Timer)weakReference.Target;

        // Will always return true, since this is only called from the Timer callback, which will have a strong reference to the Timer and the CTS
        return timer != null;
    }
#else
    private static bool TryAddTimer(this ConditionalWeakTable<CancellationTokenSource, Timer> table, CancellationTokenSource cancelSource, Timer timer)
    {
        try
        {
            table.Add(cancelSource, timer);

            return true;
        }
        catch (ArgumentException)
        {
            return false;
        }
    }

    private static bool TryGetTimer(this ConditionalWeakTable<CancellationTokenSource, Timer> table, CancellationTokenSource cancelSource, out Timer timer)
    {
        return table.TryGetValue(cancelSource, out timer);
    }

    private static bool TryRemoveTimer(this ConditionalWeakTable<CancellationTokenSource, Timer> table, CancellationTokenSource cancelSource, out Timer timer)
    {
        if (!table.TryGetValue(cancelSource, out timer))
            return false;

        table.Remove(cancelSource);
        return true;
    }
#endif

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task"/> completes.
    /// </summary>
    /// <param name="task">The target Task</param>
    /// <param name="continuationAction">
    /// An action to run when the <see cref="Task"/> completes. When run, the delegate will be
    /// passed the completed task as and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation action.</param>
    /// <returns>A new continuation <see cref="Task"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task"/> will not be scheduled for execution until the current task has
    /// completed, whether it completes due to running to completion successfully, faulting due to an
    /// unhandled exception, or exiting out early due to being canceled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationAction"/> argument is null.
    /// </exception>
    public static Task ContinueWith(this Task task, Action<Task, object> continuationAction, object state)
    {
        return task.ContinueWith(new ContinueWithState(continuationAction, state).ContinueWith);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task"/> completes.
    /// </summary>
    /// <param name="task">The target Task</param>
    /// <param name="continuationAction">
    /// An action to run when the <see cref="Task"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation action.</param>
    /// <param name="cancellationToken"> The <see cref="CancellationToken"/> that will be assigned to the new continuation task.</param>
    /// <returns>A new continuation <see cref="Task"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task"/> will not be scheduled for execution until the current task has
    /// completed, whether it completes due to running to completion successfully, faulting due to an
    /// unhandled exception, or exiting out early due to being canceled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationAction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ObjectDisposedException">The provided <see cref="System.Threading.CancellationToken">CancellationToken</see>
    /// has already been disposed.
    /// </exception>
    public static Task ContinueWith(this Task task, Action<Task, object> continuationAction, object state, CancellationToken cancellationToken)
    {
        return task.ContinueWith(new ContinueWithState(continuationAction, state).ContinueWith, cancellationToken);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task"/> completes.
    /// </summary>
    /// <param name="task">The target Task</param>
    /// <param name="continuationAction">
    /// An action to run when the <see cref="Task"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation action.</param>
    /// <param name="continuationOptions">
    /// Options for when the continuation is scheduled and how it behaves. This includes criteria, such
    /// as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.OnlyOnCanceled">OnlyOnCanceled</see>, as
    /// well as execution options, such as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.ExecuteSynchronously">ExecuteSynchronously</see>.
    /// </param>
    /// <returns>A new continuation <see cref="Task"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task"/> will not be scheduled for execution until the current task has
    /// completed. If the continuation criteria specified through the <paramref
    /// name="continuationOptions"/> parameter are not met, the continuation task will be canceled
    /// instead of scheduled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationAction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ArgumentOutOfRangeException">
    /// The <paramref name="continuationOptions"/> argument specifies an invalid value for <see
    /// cref="T:System.Threading.Tasks.TaskContinuationOptions">TaskContinuationOptions</see>.
    /// </exception>
    public static Task ContinueWith(this Task task, Action<Task, object> continuationAction, object state, TaskContinuationOptions continuationOptions)
    {
        return task.ContinueWith(new ContinueWithState(continuationAction, state).ContinueWith, CancellationToken.None, continuationOptions, TaskScheduler.Current);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task"/> completes.
    /// </summary>
    /// <param name="task">The target Task</param>
    /// <param name="continuationAction">
    /// An action to run when the <see cref="Task"/> completes.  When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation action.</param>
    /// <param name="scheduler">
    /// The <see cref="TaskScheduler"/> to associate with the continuation task and to use for its execution.
    /// </param>
    /// <returns>A new continuation <see cref="Task"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task"/> will not be scheduled for execution until the current task has
    /// completed, whether it completes due to running to completion successfully, faulting due to an
    /// unhandled exception, or exiting out early due to being canceled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationAction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="scheduler"/> argument is null.
    /// </exception>
    public static Task ContinueWith(this Task task, Action<Task, object> continuationAction, object state, TaskScheduler scheduler)
    {
        return task.ContinueWith(new ContinueWithState(continuationAction, state).ContinueWith, CancellationToken.None, TaskContinuationOptions.None, scheduler);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task"/> completes.
    /// </summary>
    /// <param name="task">The target Task</param>
    /// <param name="continuationAction">
    /// An action to run when the <see cref="Task"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation action.</param>
    /// <param name="continuationOptions">
    /// Options for when the continuation is scheduled and how it behaves. This includes criteria, such
    /// as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.OnlyOnCanceled">OnlyOnCanceled</see>, as
    /// well as execution options, such as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.ExecuteSynchronously">ExecuteSynchronously</see>.
    /// </param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> that will be assigned to the new continuation task.</param>
    /// <param name="scheduler">
    /// The <see cref="TaskScheduler"/> to associate with the continuation task and to use for its
    /// execution.
    /// </param>
    /// <returns>A new continuation <see cref="Task"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task"/> will not be scheduled for execution until the current task has
    /// completed. If the criteria specified through the <paramref name="continuationOptions"/> parameter
    /// are not met, the continuation task will be canceled instead of scheduled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationAction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ArgumentOutOfRangeException">
    /// The <paramref name="continuationOptions"/> argument specifies an invalid value for <see
    /// cref="T:System.Threading.Tasks.TaskContinuationOptions">TaskContinuationOptions</see>.
    /// </exception>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="scheduler"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ObjectDisposedException">The provided <see cref="System.Threading.CancellationToken">CancellationToken</see>
    /// has already been disposed.
    /// </exception>
    public static Task ContinueWith(this Task task, Action<Task, object> continuationAction, object state, CancellationToken cancellationToken, TaskContinuationOptions continuationOptions, TaskScheduler scheduler)
    {
        return task.ContinueWith(new ContinueWithState(continuationAction, state).ContinueWith, cancellationToken, continuationOptions, scheduler);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task{TResult}"/> completes.
    /// </summary>
    /// <typeparam name="TResult">The type of result from the target task</typeparam>
    /// <param name="task">The target Task</param>
    /// <param name="continuationAction">
    /// An action to run when the <see cref="Task{TResult}"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation action.</param>
    /// <returns>A new continuation <see cref="Task"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task"/> will not be scheduled for execution until the current task has
    /// completed, whether it completes due to running to completion successfully, faulting due to an
    /// unhandled exception, or exiting out early due to being canceled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationAction"/> argument is null.
    /// </exception>
    public static Task ContinueWith<TResult>(this Task<TResult> task, Action<Task<TResult>, object> continuationAction, object state)
    {
        return task.ContinueWith(new ContinueWithInState<TResult>(continuationAction, state).ContinueWith);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task{TResult}"/> completes.
    /// </summary>
    /// <typeparam name="TResult">The type of result from the target task</typeparam>
    /// <param name="task">The target Task</param>
    /// <param name="continuationAction">
    /// An action to run when the <see cref="Task{TResult}"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation action.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> that will be assigned to the new continuation task.</param>
    /// <returns>A new continuation <see cref="Task"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task"/> will not be scheduled for execution until the current task has
    /// completed, whether it completes due to running to completion successfully, faulting due to an
    /// unhandled exception, or exiting out early due to being canceled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationAction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ObjectDisposedException">The provided <see cref="System.Threading.CancellationToken">CancellationToken</see>
    /// has already been disposed.
    /// </exception>
    public static Task ContinueWith<TResult>(this Task<TResult> task, Action<Task<TResult>, object> continuationAction, object state, CancellationToken cancellationToken)
    {
        return task.ContinueWith(new ContinueWithInState<TResult>(continuationAction, state).ContinueWith, cancellationToken);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task{TResult}"/> completes.
    /// </summary>
    /// <typeparam name="TResult">The type of result from the target task</typeparam>
    /// <param name="task">The target Task</param>
    /// <param name="continuationAction">
    /// An action to run when the <see cref="Task{TResult}"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation action.</param>
    /// <param name="continuationOptions">
    /// Options for when the continuation is scheduled and how it behaves. This includes criteria, such
    /// as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.OnlyOnCanceled">OnlyOnCanceled</see>, as
    /// well as execution options, such as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.ExecuteSynchronously">ExecuteSynchronously</see>.
    /// </param>
    /// <returns>A new continuation <see cref="Task"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task"/> will not be scheduled for execution until the current task has
    /// completed. If the continuation criteria specified through the <paramref
    /// name="continuationOptions"/> parameter are not met, the continuation task will be canceled
    /// instead of scheduled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationAction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ArgumentOutOfRangeException">
    /// The <paramref name="continuationOptions"/> argument specifies an invalid value for <see
    /// cref="T:System.Threading.Tasks.TaskContinuationOptions">TaskContinuationOptions</see>.
    /// </exception>
    public static Task ContinueWith<TResult>(this Task<TResult> task, Action<Task<TResult>, object> continuationAction, object state, TaskContinuationOptions continuationOptions)
    {
        return task.ContinueWith(new ContinueWithInState<TResult>(continuationAction, state).ContinueWith, CancellationToken.None, continuationOptions, TaskScheduler.Current);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task{TResult}"/> completes.
    /// </summary>
    /// <typeparam name="TResult">The type of result from the target task</typeparam>
    /// <param name="task">The target Task</param>
    /// <param name="continuationAction">
    /// An action to run when the <see cref="Task{TResult}"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation action.</param>
    /// <param name="scheduler">
    /// The <see cref="TaskScheduler"/> to associate with the continuation task and to use for its execution.
    /// </param>
    /// <returns>A new continuation <see cref="Task"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task"/> will not be scheduled for execution until the current task has
    /// completed, whether it completes due to running to completion successfully, faulting due to an
    /// unhandled exception, or exiting out early due to being canceled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationAction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="scheduler"/> argument is null.
    /// </exception>
    public static Task ContinueWith<TResult>(this Task<TResult> task, Action<Task<TResult>, object> continuationAction, object state, TaskScheduler scheduler)
    {
        return task.ContinueWith(new ContinueWithInState<TResult>(continuationAction, state).ContinueWith, CancellationToken.None, TaskContinuationOptions.None, scheduler);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task{TResult}"/> completes.
    /// </summary>
    /// <typeparam name="TResult">The type of result from the target task</typeparam>
    /// <param name="task">The target Task</param>
    /// <param name="continuationAction">
    /// An action to run when the <see cref="Task{TResult}"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation action.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> that will be assigned to the new continuation task.</param>
    /// <param name="continuationOptions">
    /// Options for when the continuation is scheduled and how it behaves. This includes criteria, such
    /// as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.OnlyOnCanceled">OnlyOnCanceled</see>, as
    /// well as execution options, such as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.ExecuteSynchronously">ExecuteSynchronously</see>.
    /// </param>
    /// <param name="scheduler">
    /// The <see cref="TaskScheduler"/> to associate with the continuation task and to use for its
    /// execution.
    /// </param>
    /// <returns>A new continuation <see cref="Task"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task"/> will not be scheduled for execution until the current task has
    /// completed. If the criteria specified through the <paramref name="continuationOptions"/> parameter
    /// are not met, the continuation task will be canceled instead of scheduled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationAction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ArgumentOutOfRangeException">
    /// The <paramref name="continuationOptions"/> argument specifies an invalid value for <see
    /// cref="T:System.Threading.Tasks.TaskContinuationOptions">TaskContinuationOptions</see>.
    /// </exception>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="scheduler"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ObjectDisposedException">The provided <see cref="System.Threading.CancellationToken">CancellationToken</see>
    /// has already been disposed.
    /// </exception>
    public static Task ContinueWith<TResult>(this Task<TResult> task, Action<Task<TResult>, object> continuationAction, object state, CancellationToken cancellationToken, TaskContinuationOptions continuationOptions, TaskScheduler scheduler)
    {
        return task.ContinueWith(new ContinueWithInState<TResult>(continuationAction, state).ContinueWith, cancellationToken, continuationOptions, scheduler);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task"/> completes.
    /// </summary>
    /// <typeparam name="TResult">
    /// The type of the result produced by the continuation.
    /// </typeparam>
    /// <param name="task">The target Task</param>
    /// <param name="continuationFunction">
    /// A function to run when the <see cref="Task"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation function.</param>
    /// <returns>A new continuation <see cref="Task{TResult}"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task{TResult}"/> will not be scheduled for execution until the current task has
    /// completed, whether it completes due to running to completion successfully, faulting due to an
    /// unhandled exception, or exiting out early due to being canceled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationFunction"/> argument is null.
    /// </exception>
    public static Task<TResult> ContinueWith<TResult>(this Task task, Func<Task, object, TResult> continuationFunction, object state)
    {
        return task.ContinueWith(new ContinueWithOutState<TResult>(continuationFunction, state).ContinueWith);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task"/> completes.
    /// </summary>
    /// <typeparam name="TResult">
    /// The type of the result produced by the continuation.
    /// </typeparam>
    /// <param name="task">The target Task</param>
    /// <param name="continuationFunction">
    /// A function to run when the <see cref="Task"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation function.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> that will be assigned to the new continuation task.</param>
    /// <returns>A new continuation <see cref="Task{TResult}"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task{TResult}"/> will not be scheduled for execution until the current task has
    /// completed, whether it completes due to running to completion successfully, faulting due to an
    /// unhandled exception, or exiting out early due to being canceled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationFunction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ObjectDisposedException">The provided <see cref="System.Threading.CancellationToken">CancellationToken</see>
    /// has already been disposed.
    /// </exception>
    public static Task<TResult> ContinueWith<TResult>(this Task task, Func<Task, object, TResult> continuationFunction, object state, CancellationToken cancellationToken)
    {
        return task.ContinueWith(new ContinueWithOutState<TResult>(continuationFunction, state).ContinueWith, cancellationToken);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task"/> completes.
    /// </summary>
    /// <typeparam name="TResult">
    /// The type of the result produced by the continuation.
    /// </typeparam>
    /// <param name="task">The target Task</param>
    /// <param name="continuationFunction">
    /// A function to run when the <see cref="Task"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation function.</param>
    /// <param name="continuationOptions">
    /// Options for when the continuation is scheduled and how it behaves. This includes criteria, such
    /// as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.OnlyOnCanceled">OnlyOnCanceled</see>, as
    /// well as execution options, such as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.ExecuteSynchronously">ExecuteSynchronously</see>.
    /// </param>
    /// <returns>A new continuation <see cref="Task{TResult}"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task{TResult}"/> will not be scheduled for execution until the current task has
    /// completed. If the continuation criteria specified through the <paramref
    /// name="continuationOptions"/> parameter are not met, the continuation task will be canceled
    /// instead of scheduled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationFunction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ArgumentOutOfRangeException">
    /// The <paramref name="continuationOptions"/> argument specifies an invalid value for <see
    /// cref="T:System.Threading.Tasks.TaskContinuationOptions">TaskContinuationOptions</see>.
    /// </exception>
    public static Task<TResult> ContinueWith<TResult>(this Task task, Func<Task, object, TResult> continuationFunction, object state, TaskContinuationOptions continuationOptions)
    {
        return task.ContinueWith(new ContinueWithOutState<TResult>(continuationFunction, state).ContinueWith, CancellationToken.None, continuationOptions, TaskScheduler.Current);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task"/> completes.
    /// </summary>
    /// <typeparam name="TResult">
    /// The type of the result produced by the continuation.
    /// </typeparam>
    /// <param name="task">The target Task</param>
    /// <param name="continuationFunction">
    /// A function to run when the <see cref="Task"/> completes.  When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation function.</param>
    /// <param name="scheduler">
    /// The <see cref="TaskScheduler"/> to associate with the continuation task and to use for its execution.
    /// </param>
    /// <returns>A new continuation <see cref="Task{TResult}"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task{TResult}"/> will not be scheduled for execution until the current task has
    /// completed, whether it completes due to running to completion successfully, faulting due to an
    /// unhandled exception, or exiting out early due to being canceled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationFunction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="scheduler"/> argument is null.
    /// </exception>
    public static Task<TResult> ContinueWith<TResult>(this Task task, Func<Task, object, TResult> continuationFunction, object state, TaskScheduler scheduler)
    {
        return task.ContinueWith(new ContinueWithOutState<TResult>(continuationFunction, state).ContinueWith, CancellationToken.None, TaskContinuationOptions.None, scheduler);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task"/> completes.
    /// </summary>
    /// <typeparam name="TResult">
    /// The type of the result produced by the continuation.
    /// </typeparam>
    /// <param name="task">The target Task</param>
    /// <param name="continuationFunction">
    /// A function to run when the <see cref="Task"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation function.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> that will be assigned to the new continuation task.</param>
    /// <param name="continuationOptions">
    /// Options for when the continuation is scheduled and how it behaves. This includes criteria, such
    /// as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.OnlyOnCanceled">OnlyOnCanceled</see>, as
    /// well as execution options, such as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.ExecuteSynchronously">ExecuteSynchronously</see>.
    /// </param>
    /// <param name="scheduler">
    /// The <see cref="TaskScheduler"/> to associate with the continuation task and to use for its
    /// execution.
    /// </param>
    /// <returns>A new continuation <see cref="Task{TResult}"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task{TResult}"/> will not be scheduled for execution until the current task has
    /// completed. If the criteria specified through the <paramref name="continuationOptions"/> parameter
    /// are not met, the continuation task will be canceled instead of scheduled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationFunction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ArgumentOutOfRangeException">
    /// The <paramref name="continuationOptions"/> argument specifies an invalid value for <see
    /// cref="T:System.Threading.Tasks.TaskContinuationOptions">TaskContinuationOptions</see>.
    /// </exception>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="scheduler"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ObjectDisposedException">The provided <see cref="System.Threading.CancellationToken">CancellationToken</see>
    /// has already been disposed.
    /// </exception>
    public static Task<TResult> ContinueWith<TResult>(this Task task, Func<Task, object, TResult> continuationFunction, object state, CancellationToken cancellationToken, TaskContinuationOptions continuationOptions, TaskScheduler scheduler)
    {
        return task.ContinueWith(new ContinueWithOutState<TResult>(continuationFunction, state).ContinueWith, cancellationToken, continuationOptions, scheduler);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task{TResult}"/> completes.
    /// </summary>
    /// <typeparam name="TResult">The type of the result produced by the target Task</typeparam>
    /// <typeparam name="TNewResult">
    /// The type of the result produced by the continuation.
    /// </typeparam>
    /// <param name="task">The target Task</param>
    /// <param name="continuationFunction">
    /// A function to run when the <see cref="Task{TResult}"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation function.</param>
    /// <returns>A new continuation <see cref="Task{TNewResult}"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task{TNewResult}"/> will not be scheduled for execution until the current
    /// task has completed, whether it completes due to running to completion successfully, faulting due
    /// to an unhandled exception, or exiting out early due to being canceled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationFunction"/> argument is null.
    /// </exception>
    public static Task<TNewResult> ContinueWith<TResult, TNewResult>(this Task<TResult> task, Func<Task<TResult>, object, TNewResult> continuationFunction, object state)
    {
        return task.ContinueWith(new ContinueWithInOutState<TResult, TNewResult>(continuationFunction, state).ContinueWith);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task{TResult}"/> completes.
    /// </summary>
    /// <typeparam name="TResult">The type of the result produced by the target Task</typeparam>
    /// <typeparam name="TNewResult">
    /// The type of the result produced by the continuation.
    /// </typeparam>
    /// <param name="task">The target Task</param>
    /// <param name="continuationFunction">
    /// A function to run when the <see cref="Task{TResult}"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation function.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> that will be assigned to the new task.</param>
    /// <returns>A new continuation <see cref="Task{TNewResult}"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task{TNewResult}"/> will not be scheduled for execution until the current
    /// task has completed, whether it completes due to running to completion successfully, faulting due
    /// to an unhandled exception, or exiting out early due to being canceled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationFunction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ObjectDisposedException">The provided <see cref="System.Threading.CancellationToken">CancellationToken</see>
    /// has already been disposed.
    /// </exception>
    public static Task<TNewResult> ContinueWith<TResult, TNewResult>(this Task<TResult> task, Func<Task<TResult>, object, TNewResult> continuationFunction, object state, CancellationToken cancellationToken)
    {
        return task.ContinueWith(new ContinueWithInOutState<TResult, TNewResult>(continuationFunction, state).ContinueWith, cancellationToken);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task{TResult}"/> completes.
    /// </summary>
    /// <typeparam name="TResult">The type of the result produced by the target Task</typeparam>
    /// <typeparam name="TNewResult">
    /// The type of the result produced by the continuation.
    /// </typeparam>
    /// <param name="task">The target Task</param>
    /// <param name="continuationFunction">
    /// A function to run when the <see cref="Task{TResult}"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation function.</param>
    /// <param name="continuationOptions">
    /// Options for when the continuation is scheduled and how it behaves. This includes criteria, such
    /// as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.OnlyOnCanceled">OnlyOnCanceled</see>, as
    /// well as execution options, such as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.ExecuteSynchronously">ExecuteSynchronously</see>.
    /// </param>
    /// <returns>A new continuation <see cref="Task{TNewResult}"/>.</returns>
    /// <remarks>
    /// <para>
    /// The returned <see cref="Task{TNewResult}"/> will not be scheduled for execution until the current
    /// task has completed, whether it completes due to running to completion successfully, faulting due
    /// to an unhandled exception, or exiting out early due to being canceled.
    /// </para>
    /// <para>
    /// The <paramref name="continuationFunction"/>, when executed, should return a <see
    /// cref="Task{TNewResult}"/>. This task's completion state will be transferred to the task returned
    /// from the ContinueWith call.
    /// </para>
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationFunction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ArgumentOutOfRangeException">
    /// The <paramref name="continuationOptions"/> argument specifies an invalid value for <see
    /// cref="T:System.Threading.Tasks.TaskContinuationOptions">TaskContinuationOptions</see>.
    /// </exception>
    public static Task<TNewResult> ContinueWith<TResult, TNewResult>(this Task<TResult> task, Func<Task<TResult>, object, TNewResult> continuationFunction, object state, TaskContinuationOptions continuationOptions)
    {
        return task.ContinueWith(new ContinueWithInOutState<TResult, TNewResult>(continuationFunction, state).ContinueWith, CancellationToken.None, continuationOptions, TaskScheduler.Current);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task{TResult}"/> completes.
    /// </summary>
    /// <typeparam name="TResult">The type of the result produced by the target Task</typeparam>
    /// <typeparam name="TNewResult">
    /// The type of the result produced by the continuation.
    /// </typeparam>
    /// <param name="task">The target Task</param>
    /// <param name="continuationFunction">
    /// A function to run when the <see cref="Task{TResult}"/> completes.  When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation function.</param>
    /// <param name="scheduler">
    /// The <see cref="TaskScheduler"/> to associate with the continuation task and to use for its execution.
    /// </param>
    /// <returns>A new continuation <see cref="Task{TNewResult}"/>.</returns>
    /// <remarks>
    /// The returned <see cref="Task{TNewResult}"/> will not be scheduled for execution until the current task has
    /// completed, whether it completes due to running to completion successfully, faulting due to an
    /// unhandled exception, or exiting out early due to being canceled.
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationFunction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="scheduler"/> argument is null.
    /// </exception>
    public static Task<TNewResult> ContinueWith<TResult, TNewResult>(this Task<TResult> task, Func<Task<TResult>, object, TNewResult> continuationFunction, object state, TaskScheduler scheduler)
    {
        return task.ContinueWith(new ContinueWithInOutState<TResult, TNewResult>(continuationFunction, state).ContinueWith, CancellationToken.None, TaskContinuationOptions.None, scheduler);
    }

    /// <summary>
    /// Creates a continuation that executes when the target <see cref="Task{TResult}"/> completes.
    /// </summary>
    /// <typeparam name="TResult">The type of the result produced by the target Task</typeparam>
    /// <typeparam name="TNewResult">
    /// The type of the result produced by the continuation.
    /// </typeparam>
    /// <param name="task">The target Task</param>
    /// <param name="continuationFunction">
    /// A function to run when the <see cref="Task{TResult}"/> completes. When run, the delegate will be
    /// passed the completed task and the caller-supplied state object as arguments.
    /// </param>
    /// <param name="state">An object representing data to be used by the continuation function.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> that will be assigned to the new task.</param>
    /// <param name="continuationOptions">
    /// Options for when the continuation is scheduled and how it behaves. This includes criteria, such
    /// as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.OnlyOnCanceled">OnlyOnCanceled</see>, as
    /// well as execution options, such as <see
    /// cref="System.Threading.Tasks.TaskContinuationOptions.ExecuteSynchronously">ExecuteSynchronously</see>.
    /// </param>
    /// <param name="scheduler">
    /// The <see cref="TaskScheduler"/> to associate with the continuation task and to use for its
    /// execution.
    /// </param>
    /// <returns>A new continuation <see cref="Task{TNewResult}"/>.</returns>
    /// <remarks>
    /// <para>
    /// The returned <see cref="Task{TNewResult}"/> will not be scheduled for execution until the current task has
    /// completed, whether it completes due to running to completion successfully, faulting due to an
    /// unhandled exception, or exiting out early due to being canceled.
    /// </para>
    /// <para>
    /// The <paramref name="continuationFunction"/>, when executed, should return a <see cref="Task{TNewResult}"/>.
    /// This task's completion state will be transferred to the task returned from the
    /// ContinueWith call.
    /// </para>
    /// </remarks>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="continuationFunction"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ArgumentOutOfRangeException">
    /// The <paramref name="continuationOptions"/> argument specifies an invalid value for <see
    /// cref="T:System.Threading.Tasks.TaskContinuationOptions">TaskContinuationOptions</see>.
    /// </exception>
    /// <exception cref="T:System.ArgumentNullException">
    /// The <paramref name="scheduler"/> argument is null.
    /// </exception>
    /// <exception cref="T:System.ObjectDisposedException">The provided <see cref="System.Threading.CancellationToken">CancellationToken</see>
    /// has already been disposed.
    /// </exception>
    public static Task<TNewResult> ContinueWith<TResult, TNewResult>(this Task<TResult> task, Func<Task<TResult>, object, TNewResult> continuationFunction, object state, CancellationToken cancellationToken, TaskContinuationOptions continuationOptions, TaskScheduler scheduler)
    {
        return task.ContinueWith(new ContinueWithInOutState<TResult, TNewResult>(continuationFunction, state).ContinueWith, cancellationToken, continuationOptions, scheduler);
    }

    private sealed class ContinueWithState
    {
        private readonly Action<Task, object> _continuationAction;
        private readonly object _state;

        internal ContinueWithState(Action<Task, object> continuationAction, object state)
        {
            if (continuationAction == null)
                throw new ArgumentNullException(nameof(continuationAction));

            _continuationAction = continuationAction;
            _state = state;
        }

        internal void ContinueWith(Task task)
        {
            _continuationAction(task, _state);
        }
    }

    private sealed class ContinueWithInState<TIn>
    {
        private readonly Action<Task<TIn>, object> _continuationAction;
        private readonly object _state;

        internal ContinueWithInState(Action<Task<TIn>, object> continuationAction, object state)
        {
            if (continuationAction == null)
                throw new ArgumentNullException(nameof(continuationAction));

            _continuationAction = continuationAction;
            _state = state;
        }

        internal void ContinueWith(Task<TIn> task)
        {
            _continuationAction(task, _state);
        }
    }

    private sealed class ContinueWithOutState<TOut>
    {
        private readonly Func<Task, object, TOut> _continuationFunction;
        private readonly object _state;

        internal ContinueWithOutState(Func<Task, object, TOut> continuationFunction, object state)
        {
            if (continuationFunction == null)
                throw new ArgumentNullException(nameof(continuationFunction));

            _continuationFunction = continuationFunction;
            _state = state;
        }

        internal TOut ContinueWith(Task task)
        {
            return _continuationFunction(task, _state);
        }
    }

    private sealed class ContinueWithInOutState<TIn, TOut>
    {
        private readonly Func<Task<TIn>, object, TOut> _continuationFunction;
        private readonly object _state;

        internal ContinueWithInOutState(Func<Task<TIn>, object, TOut> continuationFunction, object state)
        {
            if (continuationFunction == null)
                throw new ArgumentNullException(nameof(continuationFunction));

            _continuationFunction = continuationFunction;
            _state = state;
        }

        internal TOut ContinueWith(Task<TIn> task)
        {
            return _continuationFunction(task, _state);
        }
    }
}
// ReSharper restore CheckNamespace

#endif