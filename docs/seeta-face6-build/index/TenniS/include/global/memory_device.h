//
// Created by kier on 2018/5/19.
//

#ifndef TENSORSTACK_GLOBAL_DEVICE_MEMORY_H
#define TENSORSTACK_GLOBAL_DEVICE_MEMORY_H

#include <sstream>
#include <set>
#include "utils/except.h"
#include "core/device.h"

namespace ts {
    class TS_DEBUG_API NoMemoryDeviceException : public Exception {
    public:
        explicit NoMemoryDeviceException(const DeviceType &device_type)
                : Exception(NoMemoryDeviceMessage(device_type)), m_no_memory_device_type(device_type) {
        }

        static std::string NoMemoryDeviceMessage(const DeviceType &device_type) {
            std::ostringstream oss;
            oss << "Compute device " << device_type
                << " has no memory device registered. Please call RegisterMemoryDevice firstly.";
            return oss.str();
        }

        const DeviceType &no_memory_device_type() const {
            return m_no_memory_device_type;
        }

    private:
        DeviceType m_no_memory_device_type;
    };

    class TS_DEBUG_API ComputingMemory {
    public:
        /**
         * Query memory device of compute device
         * @param compute_device_type compute device type
         * @return memory device
         * @note May throw NoMemoryDeviceException
         */
        static DeviceType Query(const DeviceType &compute_device_type);

        /**
         * Query memory device of compute device
         * @param compute_device_type compute device
         * @return memory device
         * @note May throw NoMemoryDeviceException
         * @note The returned device id is same with querying device
         */
        static MemoryDevice Query(const Device &compute_device);

        /**
         * register an map of compute device and memory device
         * @param compute_device_type the compute device, such as CPU, GPU, CUBLAS, CUDNN, etc.
         * @param memory_device_type the memory device, such as CPU and GPU.
         */
        static void Register(const DeviceType &compute_device_type,
                             const DeviceType &memory_device_type) TS_NOEXCEPT;

        /**
         * No details for this API, so DO NOT call it
         */
        static void Clear();

        /**
         * @return set of pair of compute device and memory device
         */
        static std::set<std::pair<std::string, std::string>> AllItems() TS_NOEXCEPT;
    };
}


#endif //TENSORSTACK_GLOBAL_DEVICE_MEMORY_H
