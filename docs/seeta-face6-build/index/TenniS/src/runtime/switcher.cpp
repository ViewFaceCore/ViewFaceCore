//
// Created by yang on 2020/2/26.
//

#include "global/operator_factory.h"
#include "runtime/importor.h"
#include "runtime/switcher.h"
#include "utils/cpu_info.h"
#include "utils/api.h"

#include <vector>
#include <mutex>

const std::string tennis_dll_name = "tennis";
#if TS_PLATFORM_OS_WINDOWS
const std::string tennis_avx_fma_dll = "tennis_haswell.dll";
const std::string tennis_avx_dll = "tennis_sandy_bridge.dll";
const std::string tennis_sse_dll = "tennis_pentium.dll";
#elif TS_PLATFORM_OS_LINUX
const std::string tennis_avx_fma_dll = "libtennis_haswell.so";
const std::string tennis_avx_dll = "libtennis_sandy_bridge.so";
const std::string tennis_sse_dll = "libtennis_pentium.so";
#elif TS_PLATFORM_OS_MAC || TS_PLATFORM_OS_IOS
const std::string tennis_avx_fma_dll = "libtennis_haswell.dylib";
const std::string tennis_avx_dll = "libtennis_sandy_bridge.dylib";
const std::string tennis_sse_dll = "libtennis_pentium.dylib";
#endif

namespace ts{

    typedef ts_op_creator_map* (*get_creator_map)();
    typedef void (*free_creator_map)(ts_op_creator_map*);

    typedef ts_device_context* (*ts_initial_device_context) (const ts_Device *device);
    typedef void (*free_device_context)(ts_device_context*);

    typedef void (*ts_bind_context) (ts_device_context*);

    static inline bool check_cpu_features(const std::initializer_list<CPUFeature>& features){
        for (auto &fea : features) {
            auto flag = check_cpu_feature(fea);
            if (!flag) {
                return false;
            }
        }
        return true;
    }

    static inline std::string cut_path_tail(const std::string &path, std::string &tail) {
        auto win_sep_pos = path.rfind('\\');
        auto unix_sep_pos = path.rfind('/');
        auto sep_pos = win_sep_pos;
        if (sep_pos == std::string::npos) sep_pos = unix_sep_pos;
        else if (unix_sep_pos != std::string::npos && unix_sep_pos > sep_pos) sep_pos = unix_sep_pos;
        if (sep_pos == std::string::npos) {
            tail = path;
            return std::string();
        }
        tail = path.substr(sep_pos + 1);
        return path.substr(0, sep_pos);
    }

    static inline std::string cut_name_ext(const std::string &name_ext, std::string &ext) {
        auto dot_pos = name_ext.rfind('.');
        auto sep_pos = dot_pos;
        if (sep_pos == std::string::npos) {
            ext = std::string();
            return name_ext;
        }
        ext = name_ext.substr(sep_pos + 1);
        return name_ext.substr(0, sep_pos);
    }

    static inline std::string getmodelpath(const std::string &model_name)
    {
        std::string ret;

        char sLine[2048] = { 0 };
        std::string model_named = model_name + "d";
#if TS_PLATFORM_OS_WINDOWS
        HMODULE hmodule = GetModuleHandleA(model_name.c_str());
         if(!hmodule)
         {
             hmodule = GetModuleHandleA(model_named.c_str());
             if(!hmodule)
             {
                  return "";
             }
         }

         int num = GetModuleFileNameA(hmodule, sLine, sizeof(sLine));
         std::string tmp(sLine, num);
         std::string name;
         ret = cut_path_tail(tmp, name);
         return ret;
#else
        void* pSymbol = (void*)"";
        FILE *fp;
        char *pPath;
        std::string libname = "lib" + model_name;
        std::string libnamed = libname + "d";

        fp = fopen ("/proc/self/maps", "r");
        if ( fp != NULL )
        {
            while (!feof (fp))
            {
                unsigned long start, end;

                if ( !fgets (sLine, sizeof (sLine), fp))
                    continue;
                if ( !strstr (sLine, " r-xp ") || !strchr (sLine, '/'))
                    continue;

                sscanf (sLine, "%lx-%lx ", &start, &end);
                if (pSymbol >= (void *) start && pSymbol < (void *) end)
                {
                    char *tmp;
                    size_t len;

                    pPath = strchr (sLine, '/');

                    tmp = strrchr (pPath, '\n');
                    if (tmp) *tmp = 0;

                    len = strlen (pPath);
                    if (len > 10 && strcmp (pPath + len - 10, " (deleted)") == 0)
                    {
                        tmp = pPath + len - 10;
                        *tmp = 0;
                    }

                    std::string name;
                    std::string ext;
                    ret = cut_path_tail(pPath, name);
                    name = cut_name_ext(name, ext);
                    if(name == model_name || name == libname || name == model_named || name == libnamed)
                    {
                        fclose(fp);
                        return ret;
                    }
                }
            }
            fclose (fp);
        }
#endif
        return "";
    }


    class TS_DEBUG_API Switcher{
    public:
        using self = Switcher;
        using shared = std::shared_ptr<self>;

        Switcher(){
            m_importer = std::make_shared<Importor>();
        };
        ~Switcher(){
            free();
        };

        bool auto_switch(const ComputingDevice &device){
            //switch instruction
            std::unique_lock<std::mutex> _locker(m_switcher_mutex);
            if (m_is_loaded)
                return true;
            std::vector<CPUFeature> features = {AVX, FMA};
            auto dll_dir = getmodelpath(tennis_dll_name);
            if(check_cpu_features({AVX, FMA})){
                auto path = dll_dir + "/" + tennis_avx_fma_dll;
                TS_LOG_INFO << "Load dll:" << path << " to support AVX and FMA instruction.";
                m_is_loaded = m_importer->load(path);
                if(!m_is_loaded){
                    m_is_loaded = m_importer->load(tennis_avx_fma_dll);
                    if(!m_is_loaded)
                        TS_LOG_ERROR << "Load dll failed,The current machine does not support the AVX or FMA instruction set" << eject;
                }
            }
            else if(check_cpu_features({AVX})){
                auto path = dll_dir + "/" + tennis_avx_dll;
                TS_LOG_INFO << "Load dll:" << path << " to support AVX instruction.";
                m_is_loaded = m_importer->load(path);
                if(!m_is_loaded){
                    m_is_loaded = m_importer->load(tennis_avx_dll);
                    if(!m_is_loaded)
                        TS_LOG_ERROR << "Load dll failed,The current machine does not support the AVX instruction set" << eject;
                }
            }
            else if(check_cpu_features({SSE, SSE2})){
                auto path = dll_dir + "/" + tennis_sse_dll;
                TS_LOG_INFO << "Load dll:" << path << " to support SSE instruction.";
                m_is_loaded = m_importer->load(path);
                if(!m_is_loaded){
                    m_is_loaded = m_importer->load(tennis_sse_dll);
                    if(!m_is_loaded)
                        TS_LOG_ERROR << "Load dll failed,The current machine does not support the SSE2 instruction set" << eject;
                }
            }
            else{
                TS_LOG_ERROR <<
                             "Minimum support for SSE instruction,Otherwise you need to compile a version that does not support any instruction set" << eject;
            }

            m_pre_creator_map.reset(ts_plugin_get_creator_map(), ts_plugin_free_creator_map);

            get_creator_map creator_map_fuc =
                    (get_creator_map)m_importer->get_fuc_address("ts_plugin_get_creator_map");
            free_creator_map free_creator_map_fuc =
                    (free_creator_map)m_importer->get_fuc_address("ts_plugin_free_creator_map");
            m_creator_map.reset(creator_map_fuc(), free_creator_map_fuc);
            ts_plugin_flush_creator(m_creator_map.get());

            return true;
        }

        bool is_load_dll(){
            return m_is_loaded;
        }

        Importor::shared importor(){
            return m_importer;
        }

    private:
        void free(){
//            OperatorCreator::Clear();
            ts_plugin_flush_creator(m_pre_creator_map.get());
        };

    private:
        std::shared_ptr<Importor> m_importer;
        std::shared_ptr<ts_op_creator_map> m_pre_creator_map;
        std::shared_ptr<ts_op_creator_map> m_creator_map;

        bool m_is_loaded = false;
        std::mutex m_switcher_mutex;
    };

    static Switcher& get_switcher(){
        static Switcher switcher;
        return switcher;
    }

    void SwitchControll::auto_switch(const ComputingDevice &device){
        if (m_is_loaded)
            return;
        m_is_loaded = get_switcher().auto_switch(device);
        init_context(device);
    }

    bool SwitchControll::is_load_dll(){
        return m_is_loaded;
    }

    void SwitchControll::init_context(const ComputingDevice &device){
        if(!is_load_dll()){
            TS_LOG_ERROR << "Dynamic library not loaded, please call auto_switch first" << eject;
        }

        auto& switcher = get_switcher();
        ts_initial_device_context initial_device_context_fuc =
                (ts_initial_device_context)switcher.importor()->get_fuc_address("ts_plugin_initial_device_context");
        free_device_context free_device_context_fuc =
                (free_device_context)switcher.importor()->get_fuc_address("ts_plugin_free_device_context");
        ts_Device ts_device;
        ts_device.id = device.id();
        ts_device.type = device.type().c_str();
        m_device_context.reset(initial_device_context_fuc(&ts_device), free_device_context_fuc);
    }

    void SwitchControll::bind_context() {
        if (!is_load_dll()) {
            TS_LOG_ERROR << "Dynamic library not loaded, please call auto_switch first" << eject;
        }
        if (!m_device_context) {
            TS_LOG_ERROR << "DeviceContext is nullptr, please call init_context first" << eject;
        }
        auto& switcher = get_switcher();
        ts_bind_context ts_bind_context_fuc =
                (ts_bind_context)switcher.importor()->get_fuc_address("ts_plugin_bind_device_context");
        ts_bind_context_fuc(m_device_context.get());
    }


}

