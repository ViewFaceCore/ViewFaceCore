//
// Created by kier on 2018/11/1.
//

#include "runtime/inside/thread_pool.h"
#include "utils/ctxmgr_lite_support.h"

namespace ts {

    Thread::Thread()
            : is_running(true), task(nullptr), after_task(nullptr) {
        this->core = std::thread(&Thread::operating, this);
    }

    Thread::~Thread() {
        is_running = false;
        task_cond.notify_all();
        core.join();
    }

    void Thread::run(int signet, const Thread::task_type &task, const Thread::after_task_type &after_task) {
        std::unique_lock<std::mutex> locker(task_mutex);
        this->signet = signet;
        this->task = task;
        this->after_task = after_task;
        task_cond.notify_all();
    }

    bool Thread::busy() {
        if (!task_mutex.try_lock()) return false;
        bool is_busy = task != nullptr;
        task_mutex.unlock();
        return is_busy;
    }

    void Thread::join() {
        std::unique_lock<std::mutex> locker(task_mutex);
        while (task) task_cond.wait(locker);
    }

    void Thread::operating() {
        std::unique_lock<std::mutex> locker(task_mutex);
        while (is_running) {
            while (is_running && !task) task_cond.wait(locker);
            if (!is_running) break;
            task(signet);
            if (after_task) after_task(signet);
            task = nullptr;
            after_task = nullptr;
            task_cond.notify_all();
        }
    }

    ThreadPool::ThreadPool(size_t pool_size)
            : thread_pool(pool_size) {
        for (int i = 0; i < static_cast<int>(pool_size); ++i) {
            thread_pool[i] = new Thread();
            running_core.push_back(i);   // push all cartridge into chest
        }
    }

    ThreadPool::~ThreadPool() {
        for (int i = 0; i < static_cast<int>(thread_pool.size()); ++i) {
            delete thread_pool[i];
        }
    }

    Thread *ThreadPool::run(const Thread::task_type &task) {
        if (thread_pool.size() == 0) {
            task(0);
            return nullptr;
        } else {
            int signet = load();
            Thread *cart = this->thread_pool[signet];
            cart->run(signet, task,
                      Thread::after_task_type(std::bind(&ThreadPool::recycling_thread, this, std::placeholders::_1)));
            return cart;
        }

    }

    Thread *ThreadPool::run(const Thread::task_type &task, const Thread::after_task_type &after_task) {
        if (thread_pool.size() == 0) {
            task(0);
            return nullptr;
        } else {
            int signet = load();
            Thread *cart = this->thread_pool[signet];
            cart->run(signet, task, [this, after_task](int id) -> void {
                after_task(id);
                this->recycling_thread(id);
            });
            return cart;
        }
    }

    int ThreadPool::load() {
        std::unique_lock<std::mutex> locker(running_core_mutex);
        while (this->running_core.empty()) running_core_cond.wait(locker);
        int signet = this->running_core.front();
        this->running_core.pop_front();
        return signet;
    }

    void ThreadPool::join() {
        std::unique_lock<std::mutex> locker(running_core_mutex);
        while (this->running_core.size() != this->thread_pool.size()) running_core_cond.wait(locker);
    }

    bool ThreadPool::busy() {
        if (!running_core_mutex.try_lock()) return false;
        bool is_busy = this->running_core.size() != this->thread_pool.size();
        running_core_mutex.unlock();
        return is_busy;
    }

    size_t ThreadPool::size() const {
        return thread_pool.size();
    }

    void ThreadPool::recycling_thread(int signet) {
        std::unique_lock<std::mutex> locker(running_core_mutex);
        this->running_core.push_front(signet);
        running_core_cond.notify_all();
    }
}

TS_LITE_CONTEXT(ts::ThreadPool)
