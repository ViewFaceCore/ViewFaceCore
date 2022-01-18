//
// Created by kier on 2018/11/28.
//

#ifndef TENSORSTACK_RUNTIME_INSIDE_PARALLEL_H
#define TENSORSTACK_RUNTIME_INSIDE_PARALLEL_H


#include "thread_pool.h"
#include "utils/ctxmgr.h"
#include "utils/box.h"
#include "utils/log.h"
#include <algorithm>

#include <algorithm>

// #define TS_THREAD_BLOCK_SIZE 40960

namespace ts {
    using Range = std::pair<int, int>;

    inline ThreadPool *try_parallel(int task_number) {
        if (task_number <= 1) return nullptr;
        auto gun = ctx::ptr<ThreadPool>();
        if (gun != nullptr && gun->size() > 1) return gun;
        return nullptr;
    }

    inline void parallel_run(const std::function<void(int, int, int)> &range_solver, int begin, int end, bool joinable = true) {
        auto parallel_gun = ts::try_parallel(end - begin);
        if (parallel_gun) {
            auto parallel_ranges = ts::split_bins(begin, end, int(parallel_gun->size()));
            for (auto &parallel_range : parallel_ranges) {
                parallel_gun->run([range_solver, parallel_range](int signet){
                    range_solver(signet, parallel_range.first, parallel_range.second);
                });
            }
            if (joinable) {
                parallel_gun->join();
            }
        } else {
            range_solver(0, begin, end);
        }
    }

    inline void parallel_range(const std::function<void(int, const Range &)> &range_solver, int begin, int end, bool joinable = true) {
        auto parallel_gun = ts::try_parallel(end - begin);
        if (parallel_gun) {
            auto parallel_ranges = ts::split_bins(begin, end, int(parallel_gun->size()));
            for (auto &parallel_range : parallel_ranges) {
                parallel_gun->run([range_solver, parallel_range](int signet){
                    range_solver(signet, parallel_range);
                });
            }
            if (joinable) {
                parallel_gun->join();
            }
        } else {
            range_solver(0, Range(begin, end));
        }
    }

    inline void parallel_sync() {
        auto gun = ctx::ptr<ThreadPool>();
        if (gun) gun->join();
    }

    inline size_t parallel_size() {
        auto gun = ctx::ptr<ThreadPool>();
        return gun != nullptr ? (std::max<size_t>(gun->size(), 1)) : 1;
    }
}

#if !defined(TS_DISABLE_PARALLEL)

/**
 * Parallel for loop
 * @param var_loop_value loop value name
 * @param var_loop_begin loop begin value
 * @param var_loop_end loop end value
 * @note the TS_PARALLEL_XXX block do NOT support nest
 * @note The input parameters over 3 are the closure value in parallel run
 * Usage:
 * ```
 * TS_PARALLEL_FOR_BEGIN(i, 0, 10)
 *     std::cout << i << std::endl;
 * TS_PARALLEL_FOR_END()
 * TS_PARALLEL_SYNC
 * ```
 * equal to codes:
 * ```
 * for (int i = 0; i < 10; ++i) {
 *     std::cout << i << std::endl;
 * }
 * ```
 * , but in parallel
 * @note remeber use TS_PARALLEL_SYNC after every parallel task should sync of finish
 */
#define TS_PARALLEL_FOR_BEGIN(var_loop_value, var_loop_begin, var_loop_end, ...) \
{ \
    int __ts_parallel_begin = int(var_loop_begin); \
    int __ts_parallel_end = int(var_loop_end); \
    auto __ts_parallel_solver = [&, ## __VA_ARGS__](const int __parallel_id, int begin, int end) -> void { \
        int var_loop_value = begin; \
        for (; var_loop_value < end; ++var_loop_value) { \


/**
 * @note TS_PARALLEL_FOR_END can parse an bool value, mean the parallel tasks if is joinable
 * @see TS_PARALLEL_FOR_BEGIN
 */
#define TS_PARALLEL_FOR_END() \
        } \
    }; \
    ts::parallel_run(__ts_parallel_solver, __ts_parallel_begin, __ts_parallel_end); \
}

/**
 * Parallel for range parallel
 * @param var_range_value range parallel value name, is type of Range
 * @param var_range_begin range parallel begin value
 * @param var_rnage_end range parallel end value
 * @note the TS_PARALLEL_XXX block do NOT support nest
 * @note The input parameters over 3 are the closure value in parallel run
 * Usage:
 * ```
 * TS_PARALLEL_RANGE_BEGIN(range, 0, 10)
 *     for (int i = range.first; i < range.second; ++i) {
 *         std::cout << i << std::endl;
 *     }
 * TS_PARALLEL_RANGE_END()
 * TS_PARALLEL_SYNC
 * ```
 * equal to codes:
 * ```
 * {
 *     std::pair<int, int> range(0, 10)
 *     for (int i = range.first; i < range.second; ++i) {
 *         std::cout << i << std::endl;
 *     }
 * }
 * ```
 * , but in parallel
 * @note remeber use TS_PARALLEL_SYNC after every parallel task should sync of finish
 */
#define TS_PARALLEL_RANGE_BEGIN(var_range_value, var_range_begin, var_range_end, ...) \
{ \
    int __ts_parallel_begin = int(var_range_begin); \
    int __ts_parallel_end = int(var_range_end); \
    auto __ts_parallel_solver = [&, ## __VA_ARGS__](const int __parallel_id, const ts::Range &__ts_parallel_range) -> void { \
        const auto &var_range_value = __ts_parallel_range; \


/**
 * @note TS_PARALLEL_FOR_END can parse an bool value, mean the parallel tasks if is joinable
 * @see TS_PARALLEL_FOR_BEGIN
 */
#define TS_PARALLEL_RANGE_END() \
    }; \
    ts::parallel_range(__ts_parallel_solver, __ts_parallel_begin, __ts_parallel_end); \
}

/**
 * @see TS_PARALLEL_FOR_BEGIN
 */
#define TS_PARALLEL_SYNC \
ts::parallel_sync();

#define TS_PARALLEL_SIZE \
ts::parallel_size()

#else

#define TS_PARALLEL_FOR_BEGIN(var_loop_value, var_loop_begin, var_loop_end, ...) \
{ \
    const int __parallel_id = 0; \
    (void)(__parallel_id); \
    int __ts_parallel_begin = int(var_loop_begin); \
    int __ts_parallel_end = int(var_loop_end); \
    int var_loop_value = __ts_parallel_begin; \
    for (; var_loop_value < __ts_parallel_end; ++var_loop_value) {

#define TS_PARALLEL_FOR_END(...) \
    } \
}

#define TS_PARALLEL_RANGE_BEGIN(var_range_value, var_range_begin, var_range_end, ...) \
{ \
    const int __parallel_id = 0; \
    (void)(__parallel_id); \
    int __ts_parallel_begin = int(var_range_begin); \
    int __ts_parallel_end = int(var_range_end); \
    ts::Range var_range_value(__ts_parallel_begin, __ts_parallel_end);

#define TS_PARALLEL_RANGE_END(...) \
}

#define TS_PARALLEL_SYNC ;

#define TS_PARALLEL_SIZE 1

#endif


#endif //TENSORSTACK_RUNTIME_INSIDE_PARALLEL_H
