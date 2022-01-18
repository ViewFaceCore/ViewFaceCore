//
// Created by kier on 2018/12/21.
//

#include <omp.h>
#include <cstdio>

#include "runtime/runtime.h"
#include "runtime/inside/parallel.h"
#include "kernels/common/simd.h"
#include "kernels/common/openmp.h"
#include "utils/platform.h"

#if TS_PLATFORM_OS_MAC || TS_PLATFORM_OS_IOS
#include <Accelerate/Accelerate.h>
#elif TS_PLATFORM_OS_LINUX
//#include <openblas/cblas.h>
#elif TS_PLATFORM_OS_WINDOWS && TS_PLATFORM_CC_MINGW
#include <OpenBLAS/cblas.h>
#else
#include <cblas.h>
#endif

//float dot0(const float *x, const float *y, int N) {
//    return cblas_sdot(N, x, 1, y, 1);
//}

float dot1(const float *x, const float *y, int N) {
    float sum = 0;
    int i = 0;

    for (i = 0; i < N; ++i) {
        sum += *x++ * *y++;
    }

    return sum;
}

float dot2(const float *x, const float *y, int N) {
    float sum = 0;
    int i = 0;

    for (i = 0; i < N - 3; i += 4) {
        sum += *x++ * *y++;
        sum += *x++ * *y++;
        sum += *x++ * *y++;
        sum += *x++ * *y++;
    }

    for (; i < N; ++i) {
        sum += *x++ * *y++;
    }

    return sum;
}

float dot3(const float *x, const float *y, int N) {
    std::vector<float> parallel_sum(TS_PARALLEL_SIZE, 0);
    TS_PARALLEL_RANGE_BEGIN(range, 0, N)
            const float *xx = x + range.first;
            const float *yy = y + range.first;
            const auto count = range.second - range.first;
            parallel_sum[__parallel_id] += dot2(xx, yy, count);
    TS_PARALLEL_RANGE_END()
    float sum = 0;
    for (auto value : parallel_sum) sum += value;
    return sum;
}

float dot4(const float *x, const float *y, int N) {
    float sum = 0;
#pragma omp parallel for reduction(+:sum) num_threads(ts::openmp_threads(N / 4))
    for (int i = 0; i < N; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}


float dot5(const float *x, const float *y, int N) {
    float sum = 0;
    int i = 0;

    ts::float32x4 sumx4 = 0;

    for (i = 0; i < N - 3; i += 4) {
        sumx4 += ts::float32x4(x) * ts::float32x4(y);
        x += 4;
        y += 4;
    }

    sum = ts::sum(sumx4);

    for (; i < N; ++i) {
        sum += *x++ * *y++;
    }

    return sum;
}

float dot6(const float *x, const float *y, int N) {
    std::vector<float> parallel_sum(TS_PARALLEL_SIZE, 0);
    TS_PARALLEL_RANGE_BEGIN(range, 0, N)
            const float *xx = x + range.first;
            const float *yy = y + range.first;
            const auto count = range.second - range.first;
            parallel_sum[__parallel_id] += dot5(xx, yy, count);
    TS_PARALLEL_RANGE_END()
    float sum = 0;
    for (auto value : parallel_sum) sum += value;
    return sum;
}

float dot7(const float *x, const float *y, int N) {
    float sum = 0;
    ts::float32x4 sumx4 = 0;
//#pragma omp parallel for reduction(+:sumx4) num_threads(ts::openmp_threads(N / 4))
    for (int i = 0; i < N - 3; i += 4) {
        sumx4 += ts::float32x4(&x[i]) * ts::float32x4(&y[i]);
    }

    sum = ts::sum(sumx4);

    for (int i = N / 4 * 4; i < N; ++i) {
        sum += x[i] * y[i];
    }

    return sum;
}

float dot8(const float *x, const float *y, int N) {
    float sum = 0;
#pragma omp parallel for reduction(+:sum) num_threads(ts::openmp_threads(N / 4))
    for (int i = 0; i < N - 3; i += 4) {
        ts::float32x4 sumx4 = ts::float32x4(&x[i]) * ts::float32x4(&y[i]);
        sum += ts::sum(sumx4);
    }

    for (int i = N / 4 * 4; i < N; ++i) {
        sum += x[i] * y[i];
    }

    return sum;
}

void test_loop_bottom(int top, int top_id) {
#pragma omp parallel for num_threads(1)
    for (int i = 0; i < 10; ++i) {
        printf("Top task: %2d, Top ID: %d, Bottom task: %2d, Bottom ID: %d\n", top, top_id, i, omp_get_thread_num());
    }
}

void test_loop_top() {
#pragma omp parallel for num_threads(1)
    for (int i = 0; i < 10; ++i) {
        test_loop_bottom(i, omp_get_thread_num());
    }
}

using dot_function = std::function<float(const float *, const float *, int)>;

void print_avg_time(const std::string &title, const int times, dot_function func, const float *a, const float *b, int N) {
    using namespace std::chrono;
    microseconds duration(0);

    float sum = 0;

    auto start = system_clock::now();

    for (int i = 0; i < times; ++i) {
        sum += func(a, b, N);
    }

    sum /= times;

    auto end = system_clock::now();
    duration += duration_cast<microseconds>(end - start);
    double spent = 1.0 * duration.count() / 1000;

    std::cout << title << ": sum=" << sum << ", spent=" << spent << "ms" << std::endl;
}

int main()
{
    ts::RuntimeContext runtime;
    runtime.set_computing_thread_number(4);

    ts::ctx::bind<ts::ThreadPool> _bind_thread_pool(runtime.thread_pool());
    ts::ctx::bind<ts::RuntimeContext> _bind_runtime(runtime);

    srand(4482);

    static const int times = 1000;
    static const int N = 102400;
    float a[N], b[N];
    for (int i = 0; i < N; ++i) {
        a[i] = rand() % 400 / 100.0 - 2;
        b[i] = rand() % 400 / 100.0 - 2;
    }

//    print_avg_time("BLAS        ", times, dot0, a, b, N);
    print_avg_time("Pure CPU    ", times, dot1, a, b, N);
    print_avg_time("Pure CPU(4) ", times, dot2, a, b, N);
    print_avg_time("Threads CPU ", times, dot3, a, b, N);
    print_avg_time("OpenMP CPU  ", times, dot4, a, b, N);
    print_avg_time("Pure SIMD   ", times, dot5, a, b, N);
    print_avg_time("Threads SIMD", times, dot6, a, b, N);
    print_avg_time("OpenMP SIMD ", times, dot7, a, b, N);
    print_avg_time("OpenMP SIMD2", times, dot8, a, b, N);

    return 0;
}
