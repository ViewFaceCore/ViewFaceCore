//
// Created by kier on 2018/12/21.
//

#include <runtime/runtime.h>

#include "runtime/runtime.h"

#include "utils/ctxmgr_lite_support.h"

#include <algorithm>
#include <memory/flow.h>

#ifdef TS_USE_CBLAS
#if TS_PLATFORM_OS_MAC || TS_PLATFORM_OS_IOS
#include <Accelerate/Accelerate.h>
#elif TS_PLATFORM_OS_LINUX
#include <openblas/cblas.h>
#define TS_USING_OPENBLAS
#elif TS_PLATFORM_OS_WINDOWS && TS_PLATFORM_CC_MINGW
#include <OpenBLAS/cblas.h>
#define TS_USING_OPENBLAS
#else
#include <cblas.h>
#define TS_USING_OPENBLAS
#endif
#endif

#ifdef TS_USE_OPENMP
#include <omp.h>
#endif

namespace ts {
    RuntimeContext::RuntimeContext() {
        set_computing_thread_number(4);
    }
    RuntimeContext::RuntimeContext(const MemoryDevice &device): self() {
        this->m_flow = HypeSyncMemoryController<FlowMemoryController>::Make(device, false);
        this->m_dynamic = DynamicSyncMemoryController::Make(device, false);
    }

    void RuntimeContext::set_computing_thread_number(int computing_thread_number) {
        int fixed_thread_number;
        if (computing_thread_number < 0) {
            auto max_thread_number =
#ifdef TS_USE_OPENMP
                    omp_get_num_procs();
#else
                    8;
#endif
            fixed_thread_number = max_thread_number;
        } else if (computing_thread_number == 0) {
            fixed_thread_number = 1;
        } else {
            fixed_thread_number = computing_thread_number;
        }
        this->m_computing_thread_number = fixed_thread_number;

        this->m_thread_pool = std::make_shared<ThreadPool>(fixed_thread_number);
#ifdef TS_USE_CBLAS
#ifdef TS_USING_OPENBLAS
        goto_set_num_threads(fixed_thread_number);
        openblas_set_num_threads(fixed_thread_number);
#endif
#endif
    }

    int RuntimeContext::get_computing_thread_number() const {
        return m_computing_thread_number;
    }

    RuntimeContext::self RuntimeContext::clone() const {
        self doly;
        doly.m_computing_thread_number = this->m_computing_thread_number;
        if (m_thread_pool) {
            doly.m_thread_pool = std::make_shared<ThreadPool>(this->m_thread_pool->size());
        }
        if (this->m_dynamic) {
            doly.m_dynamic = this->m_dynamic->clone();
        }
        if (this->m_flow) {
            doly.m_flow = this->m_flow->clone();
        }
        return std::move(doly);
    }

    RuntimeContext::RuntimeContext(RuntimeContext::self &&other) {
        this->operator=(std::move(other));
    }

    RuntimeContext::self &RuntimeContext::operator=(RuntimeContext::self &&other) {
        std::swap(this->m_computing_thread_number, other.m_computing_thread_number);
        std::swap(this->m_thread_pool, other.m_thread_pool);
        std::swap(this->m_dynamic, other.m_dynamic);
        std::swap(this->m_flow, other.m_flow);
        return *this;
    }

    ThreadPool &RuntimeContext::thread_pool() {
        return *this->m_thread_pool;
    }

    void RuntimeContext::bind_flow(SyncMemoryController::shared flow) {
        m_flow = std::move(flow);
    }

    void RuntimeContext::bind_dynamic(SyncMemoryController::shared dynamic) {
        m_dynamic = std::move(dynamic);
    }

    SyncMemoryController::shared RuntimeContext::flow() const {
        return m_flow;
    }

    SyncMemoryController::shared RuntimeContext::dynamic() const {
        return m_dynamic;
    }

    SyncMemoryController::shared RuntimeContext::FlowMemory() {
        auto runtime = ctx::get<RuntimeContext>();
        if (!runtime) return nullptr;
        return runtime->flow();
    }

    SyncMemoryController::shared RuntimeContext::DynamicMemory() {
        auto runtime = ctx::get<RuntimeContext>();
        if (!runtime) return nullptr;
        return runtime->dynamic();}
}

TS_LITE_CONTEXT(ts::RuntimeContext)
