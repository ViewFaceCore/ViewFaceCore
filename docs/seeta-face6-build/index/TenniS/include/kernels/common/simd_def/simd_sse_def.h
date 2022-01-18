#ifndef TENSORSTACK_KERNELS_COMMON_SIMD_DEF_SIMD_SSE_DEF_H
#define TENSORSTACK_KERNELS_COMMON_SIMD_DEF_SIMD_SSE_DEF_H

#ifdef TS_USE_SSE

#include <immintrin.h>

typedef struct __m128x2
{
    __m128 val[2];
} __m128x2;

typedef struct __m128ix2
{
    __m128i val[2];
}__m128ix2;

typedef struct __m128x3
{
    __m128 val[3];
}__m128x3;

using _simd_f32x4 = __m128;
using _simd_f32x4x2 = __m128x2;
using _simd_f32x4x3 = __m128x3;
using _simd_f32 = float;
using _simd_int32x4 = __m128i;
using _simd_int32 = int32_t;
using _simd_int32x4x2 = __m128ix2;

inline _simd_int32x4 _simd_int32x4_load(const _simd_int32* p) {
    return _mm_loadu_si128((_simd_int32x4*)p);
}

inline _simd_int32x4 _simd_int32x4_set(_simd_int32 a, _simd_int32 b, _simd_int32 c, _simd_int32 d) {
    return _mm_set_epi32(d, c, b, a);
}

inline void _simd_int32x4_store(_simd_int32 *p, _simd_int32x4 m) {
    _mm_store_si128((_simd_int32x4*)p, m);
}

inline _simd_int32x4 _simd_int32x4_add(_simd_int32x4 lhs, _simd_int32x4 rhs) {
    return _mm_add_epi32(lhs, rhs);
}

inline _simd_int32x4 _simd_int32x4_sub(_simd_int32x4 lhs, _simd_int32x4 rhs) {
    return _mm_sub_epi32(lhs, rhs);
}

inline _simd_int32x4x2 _simd_int32x4x2_load(const _simd_int32* p) {
    _simd_int32x4x2 res;
    res.val[0] = _mm_loadu_si128((_simd_int32x4*)p);
    res.val[1] = _mm_loadu_si128((_simd_int32x4*)(p + 4));
    return res;
}

inline _simd_int32x4x2 _simd_int32x4x2_set(_simd_int32 a, _simd_int32 b, _simd_int32 c, _simd_int32 d,
                                           _simd_int32 e, _simd_int32 f, _simd_int32 g, _simd_int32 h) {
    _simd_int32x4x2 res;
    res.val[0] = _mm_set_epi32(d, c, b, a);
    res.val[1] = _mm_set_epi32(h, g, f, e);
    return res;
}

inline void _simd_int32x4x2_store(_simd_int32 *p, const _simd_int32x4x2 &m) {
    _mm_storeu_si128((_simd_int32x4*)p, m.val[0]);
    _mm_storeu_si128((_simd_int32x4*)(p + 4), m.val[1]);
}

inline _simd_int32x4x2 _simd_int32x4x2_add(const _simd_int32x4x2 &lhs, const _simd_int32x4x2 &rhs) {
    _simd_int32x4x2 res;
    res.val[0] = _mm_add_epi32(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_add_epi32(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_int32x4x2 _simd_int32x4x2_sub(const _simd_int32x4x2 &lhs, const _simd_int32x4x2 &rhs) {
    _simd_int32x4x2 res;
    res.val[0] = _mm_sub_epi32(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_sub_epi32(lhs.val[1], rhs.val[1]);
    return res;
}


inline _simd_f32x4 _simd_f32x4_load(const _simd_f32 *p) {
    return _mm_loadu_ps(p);
}

inline _simd_f32x4 _simd_f32x4_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d) {
    return _mm_set_ps(d, c, b, a);
}

inline void _simd_f32x4_store(_simd_f32 *p, _simd_f32x4 m) {
    _mm_storeu_ps(p, m);
}

inline _simd_f32x4 _simd_f32x4_add(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return _mm_add_ps(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_sub(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return _mm_sub_ps(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_mul(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return _mm_mul_ps(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_div(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return _mm_div_ps(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_max(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return _mm_max_ps(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_min(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return _mm_min_ps(lhs, rhs);
}

inline void _simd_f32x4_transpose4x4(_simd_f32x4& q0, _simd_f32x4& q1, _simd_f32x4& q2, _simd_f32x4& q3) {
    _MM_TRANSPOSE4_PS(q0, q1, q2, q3);
}

inline _simd_f32x4 _simd_f32x4_fmadd(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2) {
#ifdef TS_USE_FMA
    return _mm_fmadd_ps(q0, q1, q2);
#else
    return _mm_add_ps(q2, _mm_mul_ps(q0, q1));
#endif
}

inline _simd_f32x4 _simd_f32x4_fmadd(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2, const int index) {
#ifdef TS_USE_FMA
    return _mm_fmadd_ps(q0, _mm_set1_ps(*((float*)&q1 + index)), q2);
#else
    return _mm_add_ps(q2, _mm_mul_ps(q0, _mm_set1_ps(*((float*)&q1 + index))));
#endif
}

inline _simd_f32x4 _simd_broadcast2float32x4(const _simd_f32* src) {
    return _mm_set1_ps(*src);
}

inline _simd_f32x4 _simd_f32x4_concat(const _simd_f32x4& q0, const _simd_f32x4& q1, const int index) {
    if (index == 0)
        return q0;
    float res[4];
    for (int i = index; i < 4; i++) {
        res[i - index] = *(((float*)&q0) + i);
    }
    for (int i = 0; i < index; i++) {
        res[i + 4 - index] = *(((float*)&q1) + i);
    }
    return _mm_loadu_ps(res);
}

inline _simd_f32x4 _simd_f32x4_interval_load(const _simd_f32* p, const int inc) {
    const _simd_f32* a = p;
    const _simd_f32* b = a + inc;
    const _simd_f32* c = b + inc;
    const _simd_f32* d = c + inc;
    return _mm_set_ps(*d, *c, *b, *a);
}


inline _simd_f32x4x2 _simd_f32x4x2_load(const _simd_f32 *p) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_loadu_ps(p);
    res.val[1] = _mm_loadu_ps(p + 4);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d,
                                     _simd_f32 e, _simd_f32 f, _simd_f32 g, _simd_f32 h) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_set_ps(d,c,b,a);
    res.val[1] = _mm_set_ps(h,g,f,e);
    return res;
}

inline void _simd_f32x4x2_store(_simd_f32 *p, const _simd_f32x4x2 &m) {
    _mm_storeu_ps(p, m.val[0]);
    _mm_storeu_ps(p + 4, m.val[1]);
}

inline _simd_f32x4x2 _simd_f32x4x2_add(const _simd_f32x4x2 &lhs, const _simd_f32x4x2 &rhs) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_add_ps(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_add_ps(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_sub(const _simd_f32x4x2 &lhs, const _simd_f32x4x2 &rhs) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_sub_ps(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_sub_ps(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_mul(const _simd_f32x4x2 &lhs, const _simd_f32x4x2 &rhs) {
    _simd_f32x4x2 res;
    res.val[0]  =_mm_mul_ps(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_mul_ps(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_div(const _simd_f32x4x2 &lhs, const _simd_f32x4x2 &rhs) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_div_ps(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_div_ps(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_f32x4 _simd_f32x4x2_index(const _simd_f32x4x2 &src, const int index) {
    return src.val[index];
}

inline _simd_f32x4x2 _simd_f32x4x2_fmadd(const _simd_f32x4x2 &q0, const _simd_f32x4x2 &q1, const _simd_f32x4x2 &q2) {
    _simd_f32x4x2 res;
#ifdef TS_USE_FMA
    res.val[0] = _mm_fmadd_ps(q0.val[0], q1.val[0], q2.val[0]);
    res.val[1] = _mm_fmadd_ps(q0.val[1], q1.val[1], q2.val[1]);
#else
    res.val[0] = _mm_add_ps(q2.val[0], _mm_mul_ps(q0.val[0], q1.val[0]));
    res.val[1] = _mm_add_ps(q2.val[1], _mm_mul_ps(q0.val[1], q1.val[1]));
#endif
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_interval_load(const _simd_f32* p, const int inc) {
    _simd_f32x4x2 res;
    const _simd_f32* a0 = p, *a1 = p + 1;
    const _simd_f32* b0 = a0 + inc, *b1 = a1 + inc;
    const _simd_f32* c0 = b0 + inc, *c1 = b1 + inc;
    const _simd_f32* d0 = c0 + inc, *d1 = c1 + inc;
    res.val[0] = _mm_set_ps(*d0, *c0, *b0, *a0);
    res.val[1] = _mm_set_ps(*d1, *c1, *b1, *a1);
    return res;
}

inline _simd_f32x4x3 _simd_f32x4x3_load(const _simd_f32 *p) {
    _simd_f32x4x3 res;
    res.val[0] = _mm_loadu_ps(p);
    res.val[1] = _mm_loadu_ps(p + 4);
    res.val[2] = _mm_loadu_ps(p + 8);
    return res;
}

inline _simd_f32x4x3 _simd_f32x4x3_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d,
                                       _simd_f32 e, _simd_f32 f, _simd_f32 g, _simd_f32 h,
                                       _simd_f32 i, _simd_f32 j, _simd_f32 k, _simd_f32 l){
    _simd_f32x4x3 res;
    res.val[0] = _mm_set_ps(d, c, b, a);
    res.val[1] = _mm_set_ps(h, g, f, e);
    res.val[2] = _mm_set_ps(i, j, k, l);
    return res;
}

inline void _simd_f32x4x3_store(_simd_f32 *p, const _simd_f32x4x3 &m) {
    _mm_storeu_ps(p, m.val[0]);
    _mm_storeu_ps(p + 4, m.val[1]);
    _mm_storeu_ps(p + 8, m.val[2]);
}

inline void _simd_f32x4x3_store(_simd_f32 *p, const _simd_f32x4x3 &m, int index) {
    _mm_storeu_ps(p, m.val[index]);
}

//TODO: optimze this fuc,just like vldnq_f32
inline _simd_f32x4x3 _simd_f32x4x3_interval_load(const _simd_f32 *p, int inc){
    _simd_f32x4x3 res;
    const _simd_f32* a0 = p;const _simd_f32* b0 = p+1;const _simd_f32* c0 = p+2;
    const _simd_f32* a1 = a0 + inc; const _simd_f32* b1 = b0 + inc; const _simd_f32* c1 = c0 + inc;
    const _simd_f32* a2 = a1 + inc; const _simd_f32* b2 = b1 + inc; const _simd_f32* c2 = c1 + inc;
    const _simd_f32* a3 = a2 + inc; const _simd_f32* b3 = b2 + inc; const _simd_f32* c3 = c2 + inc;
    res.val[0] = _mm_set_ps(*a3, *a2, *a1, *a0);
    res.val[1] = _mm_set_ps(*b3, *b2, *b1, *b0);
    res.val[2] = _mm_set_ps(*c3, *c2, *c1, *c0);
    return res;
}

//TODO: optimize this fuc,just like vstnq_f32
inline void _simd_f32x4x3_interval_save(_simd_f32 *p, const _simd_f32x4x3 &m){
    float* i0 = (float*)&(m.val[0]);
    float* i1 = (float*)&(m.val[1]);
    float* i2 = (float*)&(m.val[2]);
    for (int i = 0; i < 4; ++i) {
        *p++ = *i0++;
        *p++ = *i1++;
        *p++ = *i2++;
    }
}

//cast
inline _simd_int32x4x2 _simd_floatx4x2_to_int32x4x2(const _simd_f32x4x2 &src) {
    _simd_int32x4x2 res;
    res.val[0] = _mm_cvtps_epi32(src.val[0]);
    res.val[1] = _mm_cvtps_epi32(src.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_intx4x2_to_float32x4x2(const _simd_int32x4x2 &src) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_cvtepi32_ps(src.val[0]);
    res.val[1] = _mm_cvtepi32_ps(src.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_broadcast2float32x4x2(const _simd_f32* src) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_set1_ps(*src);
    res.val[1] = _mm_set1_ps(*src);
    return res;
}

//concat
inline _simd_f32x4x3 _simd_concat(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2){
    _simd_f32x4x3 res;
    res.val[0] = q0;
    res.val[1] = q1;
    res.val[2] = q2;
    return res;
}

#endif //TS_USE_SSE

#endif //TENSORSTACK_KERNELS_COMMON_SIMD_DEF_SIMD_SSE_DEF_H
