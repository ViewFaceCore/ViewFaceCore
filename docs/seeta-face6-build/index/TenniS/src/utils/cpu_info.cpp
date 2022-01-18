/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modify:Remove some unnecessary feature detection
==============================================================================*/

#include "utils/cpu_info.h"
#include "utils/assert.h"

#if TS_PLATFORM_IS_X86
#include <mutex>
#endif

#if TS_PLATFORM_IS_X86
#if TS_PLATFORM_OS_WINDOWS
// Visual Studio defines a builtin function for CPUID, so use that if possible.
#define GETCPUID(a, b, c, d, a_inp, c_inp) \
  {                                        \
    int cpu_info[4] = {-1};                \
    __cpuidex(cpu_info, a_inp, c_inp);     \
    a = cpu_info[0];                       \
    b = cpu_info[1];                       \
    c = cpu_info[2];                       \
    d = cpu_info[3];                       \
  }
#else
// Otherwise use gcc-format assembler to implement the underlying instructions.
#define GETCPUID(a, b, c, d, a_inp, c_inp) \
  asm("mov %%rbx, %%rdi\n"                 \
      "cpuid\n"                            \
      "xchg %%rdi, %%rbx\n"                \
      : "=a"(a), "=D"(b), "=c"(c), "=d"(d) \
      : "a"(a_inp), "2"(c_inp))
#endif
#endif //TS_PLATFORM_IS_X86

namespace ts {

#if TS_PLATFORM_IS_X86

    class CPUIDInfo;

    void init_cpuid_info();

    CPUIDInfo *cpuid = nullptr;

#if TS_PLATFORM_OS_WINDOWS
    // Visual Studio defines a builtin function, so use that if possible.
    int GetXCR0EAX() { return int(_xgetbv(0)); }
#else

    int GetXCR0EAX() {
        int eax, edx;
        asm("XGETBV" : "=a"(eax), "=d"(edx) : "c"(0));
        return eax;
    }

#endif

    // Structure for basic CPUID info
    class CPUIDInfo {
    public:
        CPUIDInfo()
                : have_avx_(0),
                  have_avx2_(0),
                  have_fma_(0),
                  have_sse_(0),
                  have_sse2_(0),
                  have_sse3_(0),
                  have_sse4_1_(0),
                  have_sse4_2_(0),
                  have_ssse3_(0) {}

        static void Initialize() {
            // Initialize cpuid struct
            TS_CHECK(cpuid == nullptr) << __func__ << " ran more than once";
            cpuid = new CPUIDInfo;

            uint32_t eax, ebx, ecx, edx;

            // Get vendor string (issue CPUID with eax = 0)
            GETCPUID(eax, ebx, ecx, edx, 0, 0);
            cpuid->vendor_str_.append(reinterpret_cast<char *>(&ebx), 4);
            cpuid->vendor_str_.append(reinterpret_cast<char *>(&edx), 4);
            cpuid->vendor_str_.append(reinterpret_cast<char *>(&ecx), 4);

            // To get general information and extended features we send eax = 1 and
            // ecx = 0 to cpuid.  The response is returned in eax, ebx, ecx and edx.
            // (See Intel 64 and IA-32 Architectures Software Developer's Manual
            // Volume 2A: Instruction Set Reference, A-M CPUID).
            GETCPUID(eax, ebx, ecx, edx, 1, 0);

            cpuid->have_sse2_ = (edx >> 26) & 0x1;
            cpuid->have_sse3_ = ecx & 0x1;
            cpuid->have_sse4_1_ = (ecx >> 19) & 0x1;
            cpuid->have_sse4_2_ = (ecx >> 20) & 0x1;
            cpuid->have_sse_ = (edx >> 25) & 0x1;
            cpuid->have_ssse3_ = (ecx >> 9) & 0x1;

            const uint64_t xcr0_xmm_mask = 0x2;
            const uint64_t xcr0_ymm_mask = 0x4;
            // const uint64_t xcr0_maskreg_mask = 0x20;
            // const uint64_t xcr0_zmm0_15_mask = 0x40;
            // const uint64_t xcr0_zmm16_31_mask = 0x80;

            const uint64_t xcr0_avx_mask = xcr0_xmm_mask | xcr0_ymm_mask;
            const bool have_avx =
                    // Does the OS support XGETBV instruction use by applications?
                    ((ecx >> 27) & 0x1) &&
                    // Does the OS save/restore XMM and YMM state?
                    ((GetXCR0EAX() & xcr0_avx_mask) == xcr0_avx_mask) &&
                    // Is AVX supported in hardware?
                    ((ecx >> 28) & 0x1);

            cpuid->have_avx_ = have_avx;
            cpuid->have_fma_ = have_avx && ((ecx >> 12) & 0x1);

            // Get standard level 7 structured extension features (issue CPUID with
            // eax = 7 and ecx= 0), which is required to check for AVX2 support as
            // well as other Haswell (and beyond) features.  (See Intel 64 and IA-32
            // Architectures Software Developer's Manual Volume 2A: Instruction Set
            // Reference, A-M CPUID).
            GETCPUID(eax, ebx, ecx, edx, 7, 0);

            cpuid->have_avx2_ = have_avx && ((ebx >> 5) & 0x1);

        }

        static bool check_feature(CPUFeature feature) {
            init_cpuid_info();
            // clang-format off
            switch (feature) {
                case AVX2:
                    return cpuid->have_avx2_;
                case AVX:
                    return cpuid->have_avx_;
                case FMA:
                    return cpuid->have_fma_;
                case SSE2:
                    return cpuid->have_sse2_;
                case SSE3:
                    return cpuid->have_sse3_;
                case SSE4_1:
                    return cpuid->have_sse4_1_;
                case SSE4_2:
                    return cpuid->have_sse4_2_;
                case SSE:
                    return cpuid->have_sse_;
                case SSSE3:
                    return cpuid->have_ssse3_;
                default:
                    break;
            }
            // clang-format on
            return false;
        }

    private:
        int have_avx_ : 1;
        int have_avx2_ : 1;
        int have_fma_ : 1;
        int have_sse_ : 1;
        int have_sse2_ : 1;
        int have_sse3_ : 1;
        int have_sse4_1_ : 1;
        int have_sse4_2_ : 1;
        int have_ssse3_ : 1;
        std::string vendor_str_;
    };

    std::once_flag cpuid_once_flag;

    void init_cpuid_info() {
        // This ensures that CPUIDInfo::Initialize() is called exactly
        // once regardless of how many threads concurrently call us
        std::call_once(cpuid_once_flag, CPUIDInfo::Initialize);
    }

#endif  // TS_PLATFORM_IS_X86

    bool check_cpu_feature(CPUFeature feature) {
#if TS_PLATFORM_IS_X86
        return CPUIDInfo::check_feature(feature);
#else
        return false;
#endif
    }

}