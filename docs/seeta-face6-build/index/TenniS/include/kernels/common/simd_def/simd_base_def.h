#ifndef TENSORSTACK_KERNELS_COMMON_SIMD_DEF_SIMD_BASE_DEF_H
#define TENSORSTACK_KERNELS_COMMON_SIMD_DEF_SIMD_BASE_DEF_H

#include <array>
#include <math.h>

using _simd_f32 = float;
using _simd_f32x4 = std::array<_simd_f32, 4>;
using _simd_f32x4x2 = std::array<_simd_f32, 8>;
using _simd_f32x4x3 = std::array<_simd_f32, 12>;
using _simd_int32 = int32_t;
using _simd_int32x4 = std::array<_simd_int32, 4>;
using _simd_int32x4x2 = std::array<_simd_int32, 8>;

inline _simd_int32x4 _simd_int32x4_load(const _simd_int32* p) {
    return{ p[0], p[1], p[2], p[3] };
}

inline _simd_int32x4 _simd_int32x4_set(_simd_int32 a, _simd_int32 b, _simd_int32 c, _simd_int32 d) {
    return{ a, b, c, d };
}

inline void _simd_int32x4_store(_simd_int32 *p, _simd_int32x4 m) {
    p[0] = m[0];
    p[1] = m[1];
    p[2] = m[2];
    p[3] = m[3];
}

inline _simd_int32x4 _simd_int32x4_add(_simd_int32x4 lhs, _simd_int32x4 rhs) {
    return{ lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3] };
}

inline _simd_int32x4 _simd_int32x4_sub(_simd_int32x4 lhs, _simd_int32x4 rhs) {
    return{ lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3] };
}

inline _simd_int32x4x2 _simd_int32x4x2_load(const _simd_int32* p) {
    return{ p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7] };
}

inline _simd_int32x4x2 _simd_int32x4x2_set(_simd_int32 a, _simd_int32 b, _simd_int32 c, _simd_int32 d,
    _simd_int32 e, _simd_int32 f, _simd_int32 g, _simd_int32 h) {
    return{ a, b, c, d, e, f, g, h };
}

inline void _simd_int32x4x2_store(_simd_int32 *p, _simd_int32x4x2 m) {
    p[0] = m[0]; p[1] = m[1];
    p[2] = m[2]; p[3] = m[3];
    p[4] = m[4]; p[5] = m[5];
    p[6] = m[6]; p[7] = m[7];
}

inline _simd_int32x4x2 _simd_int32x4x2_add(_simd_int32x4x2 lhs, _simd_int32x4x2 rhs) {
    return{ lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3], lhs[4] + rhs[4], lhs[5] + rhs[5], lhs[6] + rhs[6], lhs[7] + rhs[7] };
}

inline _simd_int32x4x2 _simd_int32x4x2_sub(_simd_int32x4x2 lhs, _simd_int32x4x2 rhs) {
    return{ lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3], lhs[4] - rhs[4], lhs[5] - rhs[5], lhs[6] - rhs[6], lhs[7] - rhs[7] };
}


inline _simd_f32x4 _simd_f32x4_load(const _simd_f32 *p) {
    return { p[0], p[1], p[2], p[3] };
}

inline _simd_f32x4 _simd_f32x4_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d) {
    return { a, b, c, d };
}

inline void _simd_f32x4_store(_simd_f32 *p, _simd_f32x4 m) {
    p[0] = m[0];
    p[1] = m[1];
    p[2] = m[2];
    p[3] = m[3];
}

inline _simd_f32x4 _simd_f32x4_add(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return { lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3] };
}

inline _simd_f32x4 _simd_f32x4_sub(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return { lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3] };
}

inline _simd_f32x4 _simd_f32x4_mul(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return { lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2], lhs[3] * rhs[3] };
}

inline _simd_f32x4 _simd_f32x4_div(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return { lhs[0] / rhs[0], lhs[1] / rhs[1], lhs[2] / rhs[2], lhs[3] / rhs[3] };
}

inline _simd_f32x4 _simd_f32x4_max(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return{ std::max(lhs[0],rhs[0]), std::max(lhs[1],rhs[1]), std::max(lhs[2],rhs[2]), std::max(lhs[3],rhs[3]) };
}

inline _simd_f32x4 _simd_f32x4_min(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return{ std::min(lhs[0],rhs[0]), std::min(lhs[1],rhs[1]), std::min(lhs[2],rhs[2]), std::min(lhs[3],rhs[3]) };
}

inline void _simd_f32x4_transpose4x4(_simd_f32x4& q0, _simd_f32x4& q1, _simd_f32x4& q2, _simd_f32x4& q3) {
    //TODO:optimize?
    /*
        q0[0] = q0[0]; q1[0] = q0[1]; q2[0] = q0[2]; q3[0] = q0[3];
        q0[1] = q1[0]; q1[1] = q1[1]; q2[1] = q1[2]; q3[1] = q1[3];
        q0[2] = q2[0]; q1[2] = q2[1]; q2[2] = q2[2]; q3[2] = q2[3];
        q0[3] = q3[0]; q1[3] = q3[1]; q2[3] = q3[2]; q3[3] = q3[3];   
    */
    _simd_f32 t0[4], t1[4], t2[4], t3[4];
    t0[0] = q0[0]; t1[0] = q0[1]; t2[0] = q0[2]; t3[0] = q0[3];
    t0[1] = q1[0]; t1[1] = q1[1]; t2[1] = q1[2]; t3[1] = q1[3];
    t0[2] = q2[0]; t1[2] = q2[1]; t2[2] = q2[2]; t3[2] = q2[3];
    t0[3] = q3[0]; t1[3] = q3[1]; t2[3] = q3[2]; t3[3] = q3[3];
    for (int i = 0; i < 4; i++)
    {
        q0[i] = t0[i]; q1[i] = t1[i]; q2[i] = t2[i]; q3[i] = t3[i];
    }

}

inline _simd_f32x4 _simd_f32x4_fmadd(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2) {
    return {q0[0] * q1[0] + q2[0], q0[1] * q1[1] + q2[1], q0[2] * q1[2] + q2[2], q0[3] * q1[3] + q2[3]};
}

inline _simd_f32x4 _simd_f32x4_fmadd(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2, const int index) {
    return{ q0[0] * q1[index] + q2[0], q0[1] * q1[index] + q2[1], q0[2] * q1[index] + q2[2], q0[3] * q1[index] + q2[3] };
}

inline _simd_f32x4 _simd_broadcast2float32x4(const _simd_f32* src) {
    float val = *src;
    return{ val, val, val, val };
}

inline _simd_f32x4 _simd_f32x4_concat(const _simd_f32x4& q0, const _simd_f32x4& q1, const int index) {
    if (index == 0)
        return q0;
    _simd_f32x4 res;
    for (int i = index; i < 4; i++) {
        res[i - index] = *(((float*)&q0) + i);
    }
    for (int i = 0; i < index; i++) {
        res[i + 4 - index] = *(((float*)&q1) + i);
    }
    return res;
}

inline _simd_f32x4 _simd_f32x4_interval_load(const _simd_f32* p, int inc) {
    const _simd_f32* a = p;
    const _simd_f32* b = a + inc;
    const _simd_f32* c = b + inc;
    const _simd_f32* d = c + inc;
    return{ *a, *b, *c, *d };
}


inline _simd_f32x4x2 _simd_f32x4x2_load(const _simd_f32 *p) {
    return { p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7] };
}

inline _simd_f32x4x2 _simd_f32x4x2_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d,
                                     _simd_f32 e, _simd_f32 f, _simd_f32 g, _simd_f32 h) {
    return {a, b, c, d, e, f, g, h};
}

inline void _simd_f32x4x2_store(_simd_f32 *p, _simd_f32x4x2 m) {
    p[0] = m[0];p[1] = m[1];
    p[2] = m[2];p[3] = m[3];
    p[4] = m[4];p[5] = m[5];
    p[6] = m[6];p[7] = m[7];
}

inline _simd_f32x4x2 _simd_f32x4x2_add(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    return { lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3], lhs[4] + rhs[4], lhs[5] + rhs[5], lhs[6] + rhs[6], lhs[7] + rhs[7]};
}

inline _simd_f32x4x2 _simd_f32x4x2_sub(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    return { lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3], lhs[4] - rhs[4], lhs[5] - rhs[5], lhs[6] - rhs[6], lhs[7] - rhs[7]};
}

inline _simd_f32x4x2 _simd_f32x4x2_mul(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    return { lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2], lhs[3] * rhs[3], lhs[4] * rhs[4], lhs[5] * rhs[5], lhs[6] * rhs[6], lhs[7] * rhs[7]};
}

inline _simd_f32x4x2 _simd_f32x4x2_div(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    return { lhs[0] / rhs[0], lhs[1] / rhs[1], lhs[2] / rhs[2], lhs[3] / rhs[3], lhs[4] / rhs[4], lhs[5] / rhs[5], lhs[6] / rhs[6], lhs[7] / rhs[7]};
}

inline _simd_f32x4 _simd_f32x4x2_index(_simd_f32x4x2 src, const int index) {
    switch (index)
    {
        case 0 : return {src[0], src[1], src[2], src[3]};
        case 1 : return {src[4], src[5], src[6], src[7]};
        default:
            break;
    }
    return {0.f, 0.f, 0.f, 0.f};
}

inline _simd_f32x4x2 _simd_f32x4x2_fmadd(_simd_f32x4x2 q0, _simd_f32x4x2 q1, _simd_f32x4x2 q2) {
    return {q0[0] * q1[0] + q2[0], q0[1] * q1[1] + q2[1], q0[2] * q1[2] + q2[2], q0[3] * q1[3] + q2[3], q0[4] * q1[4] + q2[4], q0[5] * q1[5] + q2[5], q0[6] * q1[6] + q2[6], q0[7] * q1[7] + q2[7]};
}

inline _simd_f32x4x2 _simd_f32x4x2_interval_load(const _simd_f32* p, const int inc) {
    const _simd_f32* a0 = p, *a1 = p + 1;
    const _simd_f32* b0 = a0 + inc, *b1 = a1 + inc;
    const _simd_f32* c0 = b0 + inc, *c1 = b1 + inc;
    const _simd_f32* d0 = c0 + inc, *d1 = c1 + inc;
    return {*a0, *b0, *c0, *d0, *a1, *b1, *c1, *d1};
}

inline _simd_f32x4x3 _simd_f32x4x3_load(const _simd_f32 *p) {
    return { p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11] };
}

inline _simd_f32x4x3 _simd_f32x4x3_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d,
                                       _simd_f32 e, _simd_f32 f, _simd_f32 g, _simd_f32 h,
                                       _simd_f32 i, _simd_f32 j, _simd_f32 k, _simd_f32 l){
    return {a, b, c, d, e, f, g, h, i, j, k, l};
}

inline void _simd_f32x4x3_store(_simd_f32 *p, _simd_f32x4x3 m) {
    p[0] = m[0];p[1] = m[1];
    p[2] = m[2];p[3] = m[3];
    p[4] = m[4];p[5] = m[5];
    p[6] = m[6];p[7] = m[7];
    p[8] = m[8];p[9] = m[9];
    p[10] = m[10];p[11] = m[11];
}

inline void _simd_f32x4x3_store(_simd_f32 *p, _simd_f32x4x3 m, int index) {
    p[0] = m[4 * index];p[1] = m[4 * index + 1];
    p[2] = m[4 * index + 2];p[3] = m[4 * index + 3];
}

inline _simd_f32x4x3 _simd_f32x4x3_interval_load(const float *p, int inc){
    const _simd_f32* a0 = p;const _simd_f32* b0 = p+1;const _simd_f32* c0 = p+2;
    const _simd_f32* a1 = a0 + inc; const _simd_f32* b1 = b0 + inc; const _simd_f32* c1 = c0 + inc;
    const _simd_f32* a2 = a1 + inc; const _simd_f32* b2 = b1 + inc; const _simd_f32* c2 = c1 + inc;
    const _simd_f32* a3 = a2 + inc; const _simd_f32* b3 = b2 + inc; const _simd_f32* c3 = c2 + inc;
    return {*a0, *a1, *a2, *a3, *b0, *b1, *b2, *b3, *c0, *c1, *c2, *c3};
}

inline void _simd_f32x4x3_interval_save(_simd_f32 *p, _simd_f32x4x3 m){
    for (int i = 0; i < 4; ++i) {
        *p++ = m[i];
        *p++ = m[i + 4];
        *p++ = m[i + 8];
    }
}

//cast
inline _simd_int32x4x2 _simd_floatx4x2_to_int32x4x2(_simd_f32x4x2 src) {
    return{ (int32_t)round(src[0]), (int32_t)round(src[1]), (int32_t)round(src[2]), (int32_t)round(src[3]),(int32_t)round(src[4]), (int32_t)round(src[5]), (int32_t)round(src[6]), (int32_t)round(src[7]) };
}

inline _simd_f32x4x2 _simd_intx4x2_to_float32x4x2(_simd_int32x4x2 src) {
    return{ (float)src[0], (float)src[1], (float)src[2], (float)src[3],(float)src[4], (float)src[5], (float)src[6], (float)src[7] };
}

//broad cast
inline _simd_f32x4x2 _simd_broadcast2float32x4x2(const _simd_f32* src) {
    float val = *src;
    return{ val, val, val, val, val, val, val, val };
}

//concat
inline _simd_f32x4x3 _simd_concat(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2){
    _simd_f32x4x3 res;
    for (int i = 0; i < 4; ++i) {
        res[i] = q0[i];
    }
    for (int i = 0; i < 4; ++i) {
        res[i + 4] = q1[i];
    }
    for (int i = 0; i < 4; ++i) {
        res[i + 8] = q2[i];
    }
    return res;
}

#endif //TENSORSTACK_KERNELS_COMMON_SIMD_DEF_SIMD_BASE_DEF_H