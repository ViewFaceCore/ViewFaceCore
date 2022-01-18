//
// Created by yang on 2020/2/21.
//

#ifndef TENSORSTACK_RUNTIME_IMPORTOR_H
#define TENSORSTACK_RUNTIME_IMPORTOR_H

#include "utils/api.h"

#include <string>
#include <memory>

#if TS_PLATFORM_OS_WINDOWS
#include "Windows.h"
#ifdef VOID
#undef VOID
#endif
#define Handle HMODULE
#define LOAD_LIBRARY(x) LoadLibrary(x)
#define GET_FUC_ADDRESS GetProcAddress
#define GET_LAST_ERROR GetLastError
#define FREE_LIBRARY FreeLibrary
#elif TS_PLATFORM_OS_LINUX
#include <dlfcn.h>
#define Handle void*
#define LOAD_LIBRARY(x) dlopen(x,RTLD_LAZY)
#define GET_FUC_ADDRESS dlsym
#define GET_LAST_ERROR dlerror
#define FREE_LIBRARY dlclose
#elif TS_PLATFORM_OS_MAC || TS_PLATFORM_OS_IOS
#include <dlfcn.h>
#define Handle void*
#define LOAD_LIBRARY(x) dlopen(x,RTLD_LAZY)
#define GET_FUC_ADDRESS dlsym
#define GET_LAST_ERROR dlerror
#define FREE_LIBRARY dlclose
#endif

namespace ts{
    class TS_DEBUG_API Importor{
    public:
        using self = Importor;
        using shared = std::shared_ptr<self>;

        Importor() = default;

        bool load(const std::string& dll_name);
        void unload();
        void* get_fuc_address(const std::string& fuc_name);

    private:
        Handle m_handle = nullptr;
    };
}

#endif //TENSORSTACK_RUNTIME_IMPORTOR_H
