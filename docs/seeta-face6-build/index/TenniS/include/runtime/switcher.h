//
// Created by yang on 2020/2/26.
//

#ifndef TENSORSTACK_RUNTIME_SWITCHER_H
#define TENSORSTACK_RUNTIME_SWITCHER_H

#include "api/plugin.h"
#include "core/device.h"

namespace ts{

    class TS_DEBUG_API SwitchControll{
    public:
        using self = SwitchControll;
        using shared = std::shared_ptr<SwitchControll>;

        SwitchControll() = default;
        ~SwitchControll() = default;

        void auto_switch(const ComputingDevice &device);
        void init_context(const ComputingDevice &device);
        void bind_context();

        bool is_load_dll();

    private:
        std::shared_ptr<ts_device_context> m_device_context;

        bool m_is_loaded = false;
    };
}

#endif //TENSORSTACK_RUNTIME_SWITCHER_H
