//ref:https://stackoverflow.com/questions/7467848/is-it-possible-to-set-affinity-with-sched-setaffinity-in-android
//ref:https://github.com/Tencent/ncnn/blob/master/src/cpu.cpp

#include "utils/cpu.h"
#include "utils/log.h"
#include <fstream>
#include <regex>
#include <vector>
#include <memory.h>

#if TS_PLATFORM_OS_ANDROID
#include <sys/syscall.h>
#include <unistd.h>
#include <stdint.h>
#endif

#ifdef TS_USE_OPENMP
#include <omp.h>
#endif

namespace ts{

    typedef struct CpuInfo{
        int id;
        float max_freq;
    }CpuInfo;

    static std::vector<int> cpu_big_ids;
    static std::vector<int> cpu_little_ids;
    static bool g_set_cpu_set_power_mode_flag = false;

    static int static_get_cpu_num()
    {
    //NOTE:only support androi and linux now
    #if TS_PLATFORM_OS_ANDROID || TS_PLATFORM_OS_LINUX
        std::ifstream fread; 
        fread.open("/proc/cpuinfo");
        if (!fread.is_open()) {
            return 1;
        }
            
        std::string line;
        int count = 0;
        while(std::getline(fread, line)){
            std::smatch result;
            std::regex pattern("^(processor)(.*)");
            if(std::regex_match(line, result, pattern))
                count++;
        }
        fread.close();
        if(count < 1)
            return 1;
        return count;
    #endif
        return 1;
    }

    static int g_cpu_num = static_get_cpu_num();
    static int g_power_mode = CpuEnable::CpuPowerMode::BALANCE;

    static int get_cpu_max_freq(int cpu_id)
    {
#if TS_PLATFORM_OS_ANDROID
        std::ifstream fread;
        std::string id_str = std::to_string(cpu_id);
        int max_cpu_freq = 0;
        std::string path = "/sys/devices/system/cpu/cpu" + id_str + "/cpufreq/cpuinfo_max_freq";
        fread.open(path);
        if(fread.is_open()){
            std::string line;
            std::getline(fread, line);
            max_cpu_freq = std::stoi(line);
            fread.close();
            return max_cpu_freq;
        }

        path = "/sys/devices/system/cpu/cpu" + id_str + "/cpufreq/stats/time_in_state";
        fread.open(path);
        if(fread.is_open()){
            std::string line;
            while(std::getline(fread, line)){
                int freq = std::stoi(line.substr(0, line.find(' ')));
                max_cpu_freq = std::max(max_cpu_freq, freq);
            }
            fread.close();
            return max_cpu_freq;
        }

        fread.close();
        return max_cpu_freq;
#else
    return -1;
#endif
    }

    static void split_cpu_big_little(std::vector<int>& cpu_sorted_ids,std::vector<int>& cpu_big_ids,std::vector<int>& cpu_little_ids)
    {
        int cpu_num = g_cpu_num;
        cpu_sorted_ids.resize(cpu_num);
        std::vector<int> cpu_max_freq;
        cpu_max_freq.resize(cpu_num);
        for (int i = 0; i < cpu_max_freq.size(); i++){
            cpu_max_freq[i] = get_cpu_max_freq(i);
            cpu_sorted_ids[i] = i;
        }
        
        //big core first,simple bubble sort
        for (int i = 0; i < cpu_num; i++)
        {
            for (int j = i+1; j < cpu_num; j++)
            {
                if (cpu_max_freq[i] < cpu_max_freq[j])
                {
                    int tmp = cpu_sorted_ids[i];
                    cpu_sorted_ids[i] = cpu_sorted_ids[j];
                    cpu_sorted_ids[j] = tmp;

                    tmp = cpu_max_freq[i];
                    cpu_max_freq[i] = cpu_max_freq[j];
                    cpu_max_freq[j] = tmp;
                }
            }
        }
        
        //set big and little core
        int mid_freq = (cpu_max_freq.front() + cpu_max_freq.back()) / 2;
        for (int i = 0; i < cpu_num; i++){
            if(cpu_max_freq[i] >=  mid_freq){
                cpu_big_ids.push_back(cpu_sorted_ids[i]);
            }
            else{
                cpu_little_ids.push_back(cpu_sorted_ids[i]);
            }
        }
        
    }
    
    //ref:https://stackoverflow.com/questions/7467848/is-it-possible-to-set-affinity-with-sched-setaffinity-in-android
    static bool set_sched_affinity(const std::vector<int> cpu_ids)
    {
#if TS_PLATFORM_OS_ANDROID
        #define CPU_SETSIZE 1024
        #define __NCPUBITS  (8 * sizeof (unsigned long))
        typedef struct
        {
            unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
        } cpu_set_t;

        #define CPU_SET(cpu, cpusetp) \
        ((cpusetp)->__bits[(cpu)/__NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))

        #define CPU_ZERO(cpusetp) \
        memset((cpusetp), 0, sizeof(cpu_set_t))

        // set affinity for thread
        #ifdef __GLIBC__
            pid_t pid = syscall(SYS_gettid);
        #else
        #ifdef PI3
            pid_t pid = getpid();
        #else
            pid_t pid = gettid();
        #endif
        #endif

        cpu_set_t mask;
        CPU_ZERO(&mask);
        for (int i=0; i<cpu_ids.size(); i++)
        {
            CPU_SET(cpu_ids[i], &mask);
        }
        //TS_LOG_ERROR << "syscall begin: ";
        int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
        //TS_LOG_ERROR << "syscall end: " << syscallret;
        if (syscallret)
        {
            TS_LOG_ERROR << "syscall error,code: " << syscallret;
            return false;
        }

        return true;
#endif
        return false;
    }


    int CpuEnable::get_cpu_num()
    {
        return g_cpu_num;
    }

    int CpuEnable::get_cpu_big_num()
    {
        if (g_set_cpu_set_power_mode_flag)
            return int(cpu_big_ids.size());
        return -1;
    }

    int CpuEnable::get_cpu_little_num()
    {
        if (g_set_cpu_set_power_mode_flag)
            return int(cpu_little_ids.size());
        return -1;
    }

    bool CpuEnable::set_power_mode(CpuPowerMode mode)
    {
    //NOTE:only support androi and linux now
#if TS_PLATFORM_OS_ANDROID || TS_PLATFORM_OS_LINUX
        static std::vector<int> cpu_sorted_ids;
        if(cpu_sorted_ids.empty()){
            split_cpu_big_little(cpu_sorted_ids, cpu_big_ids, cpu_little_ids);
        }    

        std::vector<int> set_cpu_ids;
        bool flag = true;
        switch (mode)
        {
        case BALANCE:set_cpu_ids = cpu_sorted_ids; break;
        case BIGCORE:set_cpu_ids = cpu_big_ids; break;
        case LITTLECORE:set_cpu_ids = cpu_little_ids; break;    
        default:
            break;
        }   

        if((cpu_little_ids.empty() || cpu_little_ids.size() == 0) && mode != BALANCE){
            TS_LOG_ERROR << "cpu set power mode not supported";
            return false;
        }

        if (set_cpu_ids.size() == 0) {
            TS_LOG_ERROR << "cpu set is empty!";
            return false;
        }

#ifdef TS_USE_OPENMP
        int threads_num = set_cpu_ids.size();
        omp_set_num_threads(threads_num);
        std::vector<bool> flags;
        flags.resize(threads_num, false);
        #pragma omp parallel for
        for (int i = 0; i < threads_num; i++){
            flags[i] = set_sched_affinity(set_cpu_ids);
        }
        for (int i = 0; i < threads_num; i++){
            if(!flags[i]){
                flag = false;
                break;
            }
        }
#else
        flag = set_sched_affinity(set_cpu_ids);
#endif //end TS_USE_OPENMP
        if (flag) {
            g_power_mode = mode;
            g_set_cpu_set_power_mode_flag = true;
        }
        else {
            TS_LOG_ERROR << "set sched affinity failed";
        }
        return flag;
#endif
        return false;
    }

    CpuEnable::CpuPowerMode CpuEnable::get_power_mode()
    {
        return CpuPowerMode(g_power_mode);
    }
}
