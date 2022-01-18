//
// Created by yang on 2020/2/21.
//

#include "runtime/importor.h"
#include "utils/log.h"

namespace ts{
    bool Importor::load(const std::string& dll_name){
        m_handle = LOAD_LIBRARY(dll_name.c_str());
        if(m_handle == nullptr){
            unload();
//            TS_LOG_ERROR << "load dll: " << dll_name << " failed." << eject;
            return false;
        }
        return true;
    }

    void Importor::unload(){
        if (m_handle != nullptr) {
            FREE_LIBRARY(m_handle);
            m_handle = nullptr;
        }
    }

    void* Importor::get_fuc_address(const std::string& fuc_name){
        if(m_handle == nullptr){
            TS_LOG_ERROR << "handle is nullptr,please call load() first." << eject;
        }
        return GET_FUC_ADDRESS(m_handle, fuc_name.c_str());
    }
}


