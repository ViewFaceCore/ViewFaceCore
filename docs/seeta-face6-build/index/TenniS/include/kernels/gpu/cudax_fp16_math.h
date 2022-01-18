//
// Created by lby on 19-6-6.
//

#ifndef TENSORSTACK_CUDAX_FP16_MATH_H
#define TENSORSTACK_CUDAX_FP16_MATH_H

#ifdef TS_USE_CUDA_FP16

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cmath>

namespace ts {
// For CUDA 8
#if CUDART_VERSION < 9000
    __device__ inline half operator+(const half &lhs, const half &rhs) { return __hadd(lhs, rhs); }
    __device__ inline half operator-(const half &lhs, const half &rhs) { return __hsub(lhs, rhs); }
    __device__ inline half operator*(const half &lhs, const half &rhs) { return __hmul(lhs, rhs); }
    __device__ inline half operator/(const half &lhs, const half &rhs) { return __hdiv(lhs, rhs); }
    __device__ inline half operator-(const half &a) { return __hneg(a); }
    __device__ inline half operator==(const half &lhs, const half &rhs) { return __heq(lhs, rhs); }
    __device__ inline half operator>=(const half &lhs, const half &rhs) { return __hge(lhs, rhs); }
    __device__ inline half operator>(const half &lhs, const half &rhs) { return __hgt(lhs, rhs); }
    __device__ inline half operator<=(const half &lhs, const half &rhs) { return __hle(lhs, rhs); }
    __device__ inline half operator<(const half &lhs, const half &rhs) { return __hlt(lhs, rhs); }
    __device__ inline half operator!=(const half &lhs, const half &rhs) { return __hne(lhs, rhs); }
#endif
// For CUDA 9
// #if CUDART_VERSION < 10000
    template <typename T>
    __device__ inline T exp(const T &a) { return T(std::exp(a)); }
    __device__ inline half exp(const half &a) { return hexp(a); }
    template <typename T>
    __device__ inline T sqrt(const T &a) { return T(std::sqrt(a)); }
    __device__ inline half sqrt(const half &a) { return hsqrt(a); }
    template <typename T>
    __device__ inline T fabs(const T &a) { return T(std::fabs(a)); }
    __device__ inline half fabs(const half &a) { return a > half(0.0f) ? a : -a; }
// #endif
}

#endif

#endif //TENSORSTACK_CUDAX_FP16_MATH_H
