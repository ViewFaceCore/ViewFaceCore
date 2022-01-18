#ifndef TENSORSTACK_UTILS_CPU_H
#define TENSORSTACK_UTILS_CPU_H

#include "platform.h"
#include "utils/api.h"

namespace ts{

    class TS_DEBUG_API CpuEnable{

    public:
        enum CpuPowerMode{
            BALANCE = 0,
            BIGCORE = 1,
            LITTLECORE = 2
        };  

    public:
        CpuEnable(){}
        ~CpuEnable(){}

    public:
        static int get_cpu_num();
        static int get_cpu_big_num();
        static int get_cpu_little_num();
        static bool set_power_mode(CpuPowerMode mode);
        static CpuPowerMode get_power_mode();
        
    };
}

#endif //TENSORSTACK_UTILS_CPU_H