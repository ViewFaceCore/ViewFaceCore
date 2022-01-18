//
// Created by kier on 2018/11/28.
//

#include "runtime/inside/parallel.h"

#include "utils/log.h"

void test_parallel_for() {
    for (int i = 0; i < 10; ++i) {
        TS_PARALLEL_FOR_BEGIN(j, 0, 10, i)
                    TS_LOG_INFO << i * 10 + j;
        TS_PARALLEL_FOR_END()
    }
}

void test_parallel_range() {
    for (int i = 0; i < 10; ++i) {
        TS_PARALLEL_RANGE_BEGIN(j, i * 10, i * 10 + 10)
                    TS_LOG_INFO << "Range: [" << j.first << ", " << j.second << ")";
        TS_PARALLEL_RANGE_END()
    }
}

int main() {

    ts::ThreadPool pool(4);

    ts::ctx::bind<ts::ThreadPool> _bind(pool);

    test_parallel_for();
    test_parallel_range();

    return 0;
}

