//
// Created by seeta on 2018/7/19.
//

#include <kernels/cpu/math_cpu.h>
#include <utils/random.h>

#include <cstring>
#include <iostream>
#include <chrono>
#include <runtime/inside/thread_pool.h>
#include <utils/ctxmgr.h>

#ifdef TS_USE_CBLAS
#include <kernels/cblas/math_cblas.h>
#endif

void test_blas(ts::Random &rand, float *ratio = nullptr)
{
    using T = float;

    // ts::Random rand(4481);
    int M = rand.next(100, 1000);
    int N = rand.next(100, 1000);
    int K = rand.next(10, 100);
    std::unique_ptr<T[]> A(new T[M * K]);
    std::unique_ptr<T[]> B(new T[K * N]);
    std::unique_ptr<T[]> C(new T[M * N]);
    T alpha = rand.u() * 2;
    T beta = rand.u() * 2;
    ts::blas::Order order = rand.u() > 0.5 ? ts::blas::RowMajor : ts::blas::ColMajor;
    ts::blas::Transpose TransA = rand.u() > 0.5 ? ts::blas::NoTrans : ts::blas::Trans;
    ts::blas::Transpose TransB = rand.u() > 0.5 ? ts::blas::NoTrans : ts::blas::Trans;


    int lda, ldb, ldc;

    if (order == ts::blas::RowMajor) {
        lda = (TransA == ts::blas::NoTrans ? K : M);
        ldb = (TransB == ts::blas::NoTrans ? N : K);
        ldc = N;
    } else {
        lda = (TransA == ts::blas::Trans ? K : M);
        ldb = (TransB == ts::blas::Trans ? N : K);
        ldc = M;
    }

    for (int i = 0; i < M * K; ++i) A[i] = rand.next(-100, 100) / 10.0;
    for (int i = 0; i < K * N; ++i) B[i] = rand.next(-100, 100) / 10.0;
    for (int i = 0; i < M * N; ++i) C[i] = rand.next(-100, 100) / 10.0;
    std::unique_ptr<T[]> cblas_C(new T[M * N]);
    std::memcpy(cblas_C.get(), C.get(), M * N * sizeof(T));
    using namespace std::chrono;
    microseconds duration(0);

    auto start = system_clock::now();
    try {
        ts::cpu::math<T, T>::gemm(order, TransA, TransB, M, N, K, alpha, A.get(), lda, B.get(), ldb, beta, C.get(), ldc);
    } catch (const ts::Exception &e) {
        return;
    }
    auto end = system_clock::now();
    duration += duration_cast<microseconds>(end - start);
    double spent = 1.0 * duration.count() / 1000;

    duration = microseconds(0);
    start = system_clock::now();
#ifdef TS_USE_CBLAS
    ts::cblas::math<T>::gemm(order, TransA, TransB, M, N, K, alpha, A.get(), lda, B.get(), ldb, beta, cblas_C.get(), ldc);
#else
    ts::cpu::math<T, T>::gemm(order, TransA, TransB, M, N, K, alpha, A.get(), lda, B.get(), ldb, beta, C.get(), ldc);
#endif
    end = system_clock::now();
    duration += duration_cast<microseconds>(end - start);
    double spent2 = 1.0 * duration.count() / 1000;

    T sum = 0;
    for (int i = 0; i < M * N; ++i) sum += fabs(C[i] - cblas_C[i]);
    sum /= (M * N);

    std::cout << sum << " " << spent << "ms vs. " << spent2 << "ms, " << spent2 / spent << std::endl;
    float local_ratio = spent2 / spent;
    if (std::isnan(local_ratio) || std::isinf(local_ratio)) local_ratio = 1;
    if (ratio) *ratio = local_ratio;

}

int main() {
    using namespace ts;
    ts::ThreadPool pool(2);

    std::ostringstream oss;

    ts::ctx::bind<ts::ThreadPool> _bind_thread_pool(pool);

//    openblas_set_num_threads(16);

//    int i = 0;
//    double A[6] = { 1.0, 2.0, 1.0, -3.0, 4.0, -1.0 };
//    double B[6] = { 1.0, 2.0, 1.0, -3.0, 4.0, -1.0 };
//    double C[9] = { .5, .5, .5, .5, .5, .5, .5, .5, .5 };
//    double D[9] = { .5, .5, .5, .5, .5, .5, .5, .5, .5 };
//    double E[9] = { .5, .5, .5, .5, .5, .5, .5, .5, .5 };
//    double F[9] = { .5, .5, .5, .5, .5, .5, .5, .5, .5 };
//
//    ts::cpu::math<double>::gemm(blas::ColMajor, blas::NoTrans, blas::NoTrans, 3, 3, 2, 1, A, 3, B, 2, 1, C, 3);
    ts::Random rand(4481);
    int N = 100;
    float ratio_sum = 0;
    for (int i = 0; i < N;  ++i) {
        float ratio;
        test_blas(rand, &ratio);
        ratio_sum += ratio;
    }

    std::cout << "Finished. ratio = " << ratio_sum / N << std::flush;

    return 0;
}