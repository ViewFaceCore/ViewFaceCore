#ifndef TENSORSTACK_KERNELS_COMMON_SIMD_DEF_SIMD_NEON_DEF_H
#define TENSORSTACK_KERNELS_COMMON_SIMD_DEF_SIMD_NEON_DEF_H

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#include <utility>

using _simd_f32x4 = float32x4_t;
using _simd_f32x4x2 = float32x4x2_t;
using _simd_f32x4x3 = float32x4x3_t;
using _simd_f32x2 = float32x2_t;
using _simd_f32 = float;
using _simd_int32x4 = int32x4_t;
using _simd_int32 = int32_t;
using _simd_int32x4x2 = int32x4x2_t;

inline _simd_int32x4 _simd_int32x4_load(const _simd_int32* p) {
    return vld1q_s32(p);
}

inline _simd_int32x4 _simd_int32x4_set(_simd_int32 a, _simd_int32 b, _simd_int32 c, _simd_int32 d) {
    _simd_int32 array[4] = { a, b, c, d };
    return vld1q_s32(array);
}

inline void _simd_int32x4_store(_simd_int32 *p, _simd_int32x4 m) {
    vst1q_s32(p, m);
}

inline _simd_int32x4 _simd_int32x4_add(_simd_int32x4 lhs, _simd_int32x4 rhs) {
    return vaddq_s32(lhs, rhs);
}

inline _simd_int32x4 _simd_int32x4_sub(_simd_int32x4 lhs, _simd_int32x4 rhs) {
    return vsubq_s32(lhs, rhs);
}

inline _simd_int32x4x2 _simd_int32x4x2_load(const _simd_int32* p) {
    _simd_int32x4x2 res;
    res.val[0] = vld1q_s32(p);
    res.val[1] = vld1q_s32(p + 4);
    return std::move(res);
    //return vld2q_s32(p);
}

inline _simd_int32x4x2 _simd_int32x4x2_set(_simd_int32 a, _simd_int32 b, _simd_int32 c, _simd_int32 d,
    _simd_int32 e, _simd_int32 f, _simd_int32 g, _simd_int32 h) {
    _simd_int32x4x2 res;
    _simd_int32 array_0[4] = { a, b, c, d };
    _simd_int32 array_1[4] = { e, f, g, h };
    res.val[0] = vld1q_s32(array_0); res.val[1] = vld1q_s32(array_1);
    return std::move(res);
    //_simd_int32 array[8] = { a, b, c, d, e, f, g, h };
    //return vld2q_s32(array);
}

inline void _simd_int32x4x2_store(_simd_int32 *p, _simd_int32x4x2 m) {
    vst1q_s32(p, m.val[0]);
    vst1q_s32(p + 4, m.val[1]);
    //vst2q_s32(p, m);
}

inline _simd_int32x4x2 _simd_int32x4x2_add(_simd_int32x4x2 lhs, _simd_int32x4x2 rhs) {
    _simd_int32x4x2 res;
    res.val[0] = vaddq_s32(lhs.val[0], rhs.val[0]);
    res.val[1] = vaddq_s32(lhs.val[1], rhs.val[1]);
    return std::move(res);
}

inline _simd_int32x4x2 _simd_int32x4x2_sub(_simd_int32x4x2 lhs, _simd_int32x4x2 rhs) {
    _simd_int32x4x2 res;
    res.val[0] = vsubq_s32(lhs.val[0], rhs.val[0]);
    res.val[1] = vsubq_s32(lhs.val[1], rhs.val[1]);
    return std::move(res);
}

inline _simd_f32x4 _simd_f32x4_load(const _simd_f32 *p){
    return vld1q_f32(p);
}

inline _simd_f32x4 _simd_f32x4_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d){
    _simd_f32 array[4] = {a, b, c, d};
    return vld1q_f32(array);
}

inline void _simd_f32x4_store(_simd_f32 *p, _simd_f32x4 m){
    vst1q_f32(p, m);
}

inline _simd_f32x4 _simd_f32x4_add(_simd_f32x4 lhs, _simd_f32x4 rhs){
    return vaddq_f32(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_sub(_simd_f32x4 lhs, _simd_f32x4 rhs){
    return vsubq_f32(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_mul(_simd_f32x4 lhs, _simd_f32x4 rhs){
    return vmulq_f32(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_div(_simd_f32x4 lhs, _simd_f32x4 rhs){
    _simd_f32x4 recip = vrecpeq_f32(rhs);
    return vmulq_f32(lhs, recip);
}

inline _simd_f32x4 _simd_f32x4_max(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return vmaxq_f32(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_min(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return vminq_f32(lhs, rhs);
}

inline void _simd_f32x4_transpose4x4(_simd_f32x4& q0, _simd_f32x4& q1, _simd_f32x4& q2, _simd_f32x4& q3) {

    /*
    * q0 = (s00,s01,s02,s03)
    * q1 = (s10,s11,s12,s13)
    * q2 = (s20,s21,s22,s23)
    * q3 = (s30,s31,s32,s33)
    */
    /*
    * q01 = (s00,s10,s02,s12),(s01,s11,s03,s13)
    * q02 = (s20,s30,s22,s32),(s21,s31,s23,s33)
    */
    _simd_f32x4x2 q01 = vtrnq_f32(q0, q1);
    _simd_f32x4x2 q23 = vtrnq_f32(q2, q3);

    _simd_f32x2 d00 = vget_low_f32(q01.val[0]);
    _simd_f32x2 d01 = vget_high_f32(q01.val[0]);

    _simd_f32x2 d10 = vget_low_f32(q01.val[1]);
    _simd_f32x2 d11 = vget_high_f32(q01.val[1]);

    _simd_f32x2 d20 = vget_low_f32(q23.val[0]);
    _simd_f32x2 d21 = vget_high_f32(q23.val[0]);

    _simd_f32x2 d30 = vget_low_f32(q23.val[1]);
    _simd_f32x2 d31 = vget_high_f32(q23.val[1]);

    q0 = vcombine_f32(d00, d20);
    q1 = vcombine_f32(d10, d30);
    q2 = vcombine_f32(d01, d21);
    q3 = vcombine_f32(d11, d31);
}

inline _simd_f32x4 _simd_f32x4_fmadd(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2) {
    return vmlaq_f32(q2, q0, q1);
    //_simd_f32x4 mul_tmp = vmulq_f32(q0, q1);
    //return vaddq_f32(mul_tmp, q2);
}

//Note:index must be a constant integer,hard core.
inline _simd_f32x4 _simd_f32x4_fmadd(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2, const int index) {
#if defined(__aarch64__)
        switch (index)
        {
        case 0: return vfmaq_laneq_f32(q2, q0, q1, 0);
        case 1: return vfmaq_laneq_f32(q2, q0, q1, 1);
        case 2: return vfmaq_laneq_f32(q2, q0, q1, 2);
        case 3: return vfmaq_laneq_f32(q2, q0, q1, 3);
        default: _simd_f32 tmp[4] = { 0, 0, 0, 0 }; return vld1q_f32(tmp);
            break;
        }
#else
        switch (index)
        {
        case 0: return vmlaq_lane_f32(q2, q0, vget_low_f32(q1), 0);
        case 1: return vmlaq_lane_f32(q2, q0, vget_low_f32(q1), 1);
        case 2: return vmlaq_lane_f32(q2, q0, vget_high_f32(q1), 0);
        case 3: return vmlaq_lane_f32(q2, q0, vget_high_f32(q1), 1);
        default:_simd_f32 tmp[4] = { 0, 0, 0, 0 }; return vld1q_f32(tmp);
        }
#endif
}

inline _simd_f32x4 _simd_broadcast2float32x4(const _simd_f32* src) {
    return vdupq_n_f32(*src);
}

inline _simd_f32x4 _simd_f32x4_concat(const _simd_f32x4& q0, const _simd_f32x4& q1, const int index) {
    switch (index)
    {
    case 0: return vextq_f32(q0, q1, 0);
    case 1: return vextq_f32(q0, q1, 1);
    case 2: return vextq_f32(q0, q1, 2);
    case 3: return vextq_f32(q0, q1, 3);
    default:_simd_f32 tmp[4] = { 0, 0, 0, 0 }; return vld1q_f32(tmp);
    }
}

inline _simd_f32x4 _simd_f32x4_interval_load(const _simd_f32* p, const int inc) {
    const _simd_f32* a = p;
    const _simd_f32* b = a + inc;
    const _simd_f32* c = b + inc;
    const _simd_f32* d = c + inc;
    _simd_f32 array[4] = { *a, *b, *c, *d };
    return vld1q_f32(array);
}


inline _simd_f32x4x2 _simd_f32x4x2_load(const _simd_f32 *p) {
    _simd_f32x4x2 res;
    res.val[0] = vld1q_f32(p); 
    res.val[1] = vld1q_f32(p + 4);
    return std::move(res);
    //return vld2q_f32(p);
}

inline _simd_f32x4x2 _simd_f32x4x2_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d,
    _simd_f32 e, _simd_f32 f, _simd_f32 g, _simd_f32 h) {
    _simd_f32x4x2 res;
    _simd_f32 array_0[4] = { a, b, c, d };
    _simd_f32 array_1[4] = { e, f, g, h };
    res.val[0] = vld1q_f32(array_0); res.val[1] = vld1q_f32(array_1);
    return std::move(res);
    //_simd_f32 array[8] = { a, b, c, d, e, f, g, h };
    //return vld2q_f32(array);
}

inline void _simd_f32x4x2_store(_simd_f32 *p, _simd_f32x4x2 m) {
    vst1q_f32(p, m.val[0]);
    vst1q_f32(p + 4, m.val[1]);
    //vst2q_f32(p, m);
}

inline _simd_f32x4x2 _simd_f32x4x2_add(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    res.val[0] = vaddq_f32(lhs.val[0], rhs.val[0]);
    res.val[1] = vaddq_f32(lhs.val[1], rhs.val[1]);
    return std::move(res);
}

inline _simd_f32x4x2 _simd_f32x4x2_sub(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    res.val[0] = vsubq_f32(lhs.val[0], rhs.val[0]);
    res.val[1] = vsubq_f32(lhs.val[1], rhs.val[1]);
    return std::move(res);
}

inline _simd_f32x4x2 _simd_f32x4x2_mul(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    res.val[0] = vmulq_f32(lhs.val[0], rhs.val[0]);
    res.val[1] = vmulq_f32(lhs.val[1], rhs.val[1]);
    return std::move(res);
}

inline _simd_f32x4x2 _simd_f32x4x2_div(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    _simd_f32x4 recip_0 = vrecpeq_f32(rhs.val[0]);
    _simd_f32x4 recip_1 = vrecpeq_f32(rhs.val[1]);
    res.val[0] = vmulq_f32(lhs.val[0], recip_0);
    res.val[1] = vmulq_f32(lhs.val[1], recip_1);
    return std::move(res);
}

inline _simd_f32x4 _simd_f32x4x2_index(_simd_f32x4x2 src, const int index) {
    return src.val[index];
}

inline _simd_f32x4x2 _simd_f32x4x2_fmadd(_simd_f32x4x2 q0, _simd_f32x4x2 q1, _simd_f32x4x2 q2) {
    _simd_f32x4x2 res;
    res.val[0] = vmlaq_f32(q2.val[0], q0.val[0], q1.val[0]);
    res.val[1] = vmlaq_f32(q2.val[1], q0.val[1], q1.val[1]);
    //_simd_f32x4 mul_tmp_0 = vmulq_f32(q0.val[0], q1.val[0]);
    //_simd_f32x4 mul_tmp_1 = vmulq_f32(q0.val[1], q1.val[1]);
    //res.val[0] = vaddq_f32(mul_tmp_0, q2.val[0]);
    //res.val[1] = vaddq_f32(mul_tmp_1, q2.val[1]);
    return std::move(res);
}

inline _simd_f32x4x2 _simd_f32x4x2_interval_load(const _simd_f32* p, const int inc) {
    if(inc == 2){
        return vld2q_f32(p);
    }
    _simd_f32x4x2 res;
    const _simd_f32* a0 = p, *a1 = p + 1;
    const _simd_f32* b0 = a0 + inc, *b1 = a1 + inc;
    const _simd_f32* c0 = b0 + inc, *c1 = b1 + inc;
    const _simd_f32* d0 = c0 + inc, *d1 = c1 + inc;
    _simd_f32 array_0[4] = {*a0, *b0, *c0, *d0};
    _simd_f32 array_1[4] = {*a1, *b1, *c1, *d1};
    res.val[0] = vld1q_f32(array_0);
    res.val[1] = vld1q_f32(array_1);
    return res;
}

inline _simd_f32x4x3 _simd_f32x4x3_load(const _simd_f32 *p) {
    _simd_f32x4x3 res;
    res.val[0] = vld1q_f32(p);
    res.val[1] = vld1q_f32(p + 4);
    res.val[2] = vld1q_f32(p + 8);
    return std::move(res);
}

inline _simd_f32x4x3 _simd_f32x4x3_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d,
                                       _simd_f32 e, _simd_f32 f, _simd_f32 g, _simd_f32 h,
                                       _simd_f32 i, _simd_f32 j, _simd_f32 k, _simd_f32 l){
    _simd_f32x4x3 res;
    _simd_f32 array_0[4] = { a, b, c, d };
    _simd_f32 array_1[4] = { e, f, g, h };
    _simd_f32 array_2[4] = { i, j, k, l };
    res.val[0] = vld1q_f32(array_0);
    res.val[1] = vld1q_f32(array_1);
    res.val[2] = vld1q_f32(array_2);
    return std::move(res);
}

inline void _simd_f32x4x3_store(_simd_f32 *p, _simd_f32x4x3 m) {
    vst1q_f32(p, m.val[0]);
    vst1q_f32(p + 4, m.val[1]);
    vst1q_f32(p + 8, m.val[2]);
}

inline void _simd_f32x4x3_store(_simd_f32 *p, _simd_f32x4x3 m, int index) {
    vst1q_f32(p, m.val[index]);
}

inline _simd_f32x4x3 _simd_f32x4x3_interval_load(const float *p, int inc){
    _simd_f32x4x3 res;
    const _simd_f32* a0 = p;const _simd_f32* b0 = p+1;const _simd_f32* c0 = p+2;
    const _simd_f32* a1 = a0 + inc; const _simd_f32* b1 = b0 + inc; const _simd_f32* c1 = c0 + inc;
    const _simd_f32* a2 = a1 + inc; const _simd_f32* b2 = b1 + inc; const _simd_f32* c2 = c1 + inc;
    const _simd_f32* a3 = a2 + inc; const _simd_f32* b3 = b2 + inc; const _simd_f32* c3 = c2 + inc;
    _simd_f32 array_0[4] = { *a0, *a1, *a2, *a3 };
    _simd_f32 array_1[4] = { *b0, *b1, *b2, *b3 };
    _simd_f32 array_2[4] = { *c0, *c1, *c2, *c3 };
    res.val[0] = vld1q_f32(array_0);
    res.val[1] = vld1q_f32(array_1);
    res.val[2] = vld1q_f32(array_2);
    return res;
}

inline void _simd_f32x4x3_interval_save(_simd_f32 *p, _simd_f32x4x3 m){
    vst3q_f32(p, m);
}

//cast
inline _simd_int32x4x2 _simd_floatx4x2_to_int32x4x2(_simd_f32x4x2 src) {
    _simd_int32x4x2 res;
    res.val[0] = vcvtq_s32_f32(src.val[0]);
    res.val[1] = vcvtq_s32_f32(src.val[1]);
    return std::move(res);
}

inline _simd_f32x4x2 _simd_intx4x2_to_float32x4x2(_simd_int32x4x2 src) {
    _simd_f32x4x2 res;
    res.val[0] = vcvtq_f32_s32(src.val[0]);
    res.val[1] = vcvtq_f32_s32(src.val[1]);
    return std::move(res);
}

//broad cast
inline _simd_f32x4x2 _simd_broadcast2float32x4x2(const _simd_f32* src) {
    _simd_f32x4x2 res;
    res.val[0] = vdupq_n_f32(*src);
    res.val[1] = vdupq_n_f32(*src);
    return std::move(res);
}

//concat
inline _simd_f32x4x3 _simd_concat(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2){
    return {{q0, q1, q2}};
}

#endif

#endif //TENSORSTACK_KERNELS_COMMON_SIMD_DEF_SIMD_NEON_DEF_H