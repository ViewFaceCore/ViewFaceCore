//
// Created by kier on 2018/11/1.
//

#ifndef TENSORSTACK_RUNTIME_INSIDE_THREAD_POOL_H
#define TENSORSTACK_RUNTIME_INSIDE_THREAD_POOL_H

#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <vector>
#include <deque>
#include <memory>

#include <utils/api.h>
#include "utils/ctxmgr_lite.h"

namespace ts {
    class TS_DEBUG_API Thread {
    public:
        using self = Thread;
        using shared = std::shared_ptr<self>;

        using task_type = std::function<void(int)>;
        using after_task_type = std::function<void(int)>;

        Thread();

        ~Thread();

        Thread(const Thread &) = delete;

        const Thread &operator=(const Thread &) = delete;

        /**
         * @brief run Asynchronous build and run task, first calls the task, then calls the after_task.
         * @param signet the index to call `task(signet)` and `after_task(signet)`
         * @param task the function call in thread
         * @param after_task call it after task called
         */
        void run(int signet, const task_type &task, const after_task_type &after_task = nullptr);

        bool busy();

        void join();

    private:

        void operating();

        std::mutex task_mutex;              ///< mutex control each fire
        std::condition_variable task_cond;  ///< condition to tell if fire finished
        std::atomic<bool> is_running;       ///< object only work when dry is true

        int signet;                         ///< the argument to call `bullet(signet)` and `shell(signet)`
        task_type task = nullptr;          ///< main function call in thread
        after_task_type after_task = nullptr;   ///< side function call after `bullet` called

        std::thread core;                   ///< working thread

    };

    /**
 * @brief The ThreadPool class the thread pool
 */
    class TS_DEBUG_API ThreadPool : public SetupContext<ThreadPool> {
    public:
        using self = ThreadPool;
        using shared = std::shared_ptr<self>;

        /**
         * @brief Shotgun
         * @param pool_size The thread number in pool. Number of threads
         */
        ThreadPool(size_t pool_size);

        ~ThreadPool();

        ThreadPool(const ThreadPool &) = delete;

        const ThreadPool &operator=(const ThreadPool &) = delete;

        /**
         * @brief run Find ready thread, build task and run.
         * @param task the task ready to run
         * @return The running thread
         */
        Thread *run(const Thread::task_type &task);

        /**
         * @brief run Find ready thread, build task and run.
         * @param task the task ready to run
         * @param after_task the work after task finished
         * @return The running thread
         */
        Thread *run(const Thread::task_type &task, const Thread::after_task_type &after_task);

        /**
         * @brief join Wait all tasks working finish.
         */
        void join();

        /**
         * @brief busy Return if there are task running in thread
         * @return True if busy
         */
        bool busy();

        /**
         * @brief size Get number of threads
         * @return Number of threads
         */
        size_t size() const;

    private:

        /**
         * @brief load Get thread ready to run
         * @return Get ready thread
         */
        int load();

        /**
         * @brief recycling_thread Recycle thread
         * @param signet thread index
         */
        void recycling_thread(int signet);

        std::vector<Thread *> thread_pool;                     ///< all cartridges

        std::mutex running_core_mutex;                  ///< mutex to get cartridges
        std::condition_variable running_core_cond;      ///< active when cartridge pushed in chest
        std::deque<int> running_core;                  ///< save all cartridge ready to fire
    };
}


#endif //TENSORSTACK_RUNTIME_INSIDE_THREAD_POOL_H
