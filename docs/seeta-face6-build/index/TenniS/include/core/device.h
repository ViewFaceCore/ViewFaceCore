//
// Created by lby on 2018/3/11.
//

#ifndef TENSORSTACK_CORE_DEVICE_H
#define TENSORSTACK_CORE_DEVICE_H

#include "utils/api.h"
#include "utils/except.h"
#include "utils/otl.h"

#include <sstream>
#include <string>
#include <ostream>
#include <memory>

#include <utils/assert.h>

namespace ts {
    /**
     * DeviceType: hardware includeing CPU, GPU or other predictable device
     */
    using DeviceType = otl::string<8>;
    static const char *CPU = "cpu";
    static const char *GPU = "gpu";
    static const char *EIGEN = "eigen";
    static const char *BLAS = "blas";
    static const char *CUDNN = "cudnn";
    static const char *CUBLAS = "cublas";

    // This device and id may used in tensor view and sync, means do-not change the memory device
    static const char *PORTAL = "portal";   //
    static const int PORTAL_ID = -233;  // fake portal device

    /**
     * Device: Sepcific device
     */
    class TS_DEBUG_API Device {
    public:
        using self = Device;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        /**
         * Initialize device
         * @param type Hardware device @see Device
         * @param id Device type's id, 0, 1, or ...
         */
        Device(const DeviceType &type, int id) : m_type(type), m_id(id) {}

        /**
         * Initialize device
         * @param type Hardware device @see Device
         * @note Default id is 0
         */
        Device(const DeviceType &type) : self(type, 0) {}

        /**
         * Initialize device
         * @param type Hardware device @see Device
         * @note Default id is 0
         */
        Device(const char *type) : self(type, 0) {}

        /**
         * Initialize device like CPU:0
         */
        Device() : self(CPU, 0) {}

        /**
         * Device type
         * @return Device type
         */
        const DeviceType &type() const { return m_type; }

        /**
         * Device id
         * @return Device id
         */
        int id() const { return m_id; }

        /**
         * return repr string
         * @return repr string
         */
        const std::string repr() const { return m_type.std() + ":" + std::to_string(m_id); }

        /**
         * return string show the content
         * @return string
         */
        const std::string str() const { return m_type.std() + ":" + std::to_string(m_id); }

        static const self &portal() {
            static self _portal(PORTAL, PORTAL_ID);
            return _portal;
        }

    private:
        DeviceType m_type = CPU;  ///< Hardware device @see Device
        int m_id = 0; ///< Device type's id, 0, 1, or ...

    public:
        Device(const self &other) = default;

        Device &operator=(const self &other) = default;

        Device(self &&other) {
            *this = std::move(other);
        }

        Device &operator=(self &&other) {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
            MOVE_MEMBER(m_type);
            MOVE_MEMBER(m_id);
#undef MOVE_MEMBER
            return *this;
        }
    };

    inline std::ostream &operator<<(std::ostream &out, const Device &device) {
        TS_UNUSED(CPU);
        TS_UNUSED(GPU);
        TS_UNUSED(EIGEN);
        TS_UNUSED(BLAS);
        TS_UNUSED(CUDNN);
        TS_UNUSED(CUBLAS);
        return out << device.str();
    }

    TS_DEBUG_API bool operator==(const Device &lhs, const Device &rhs);

    TS_DEBUG_API bool operator!=(const Device &lhs, const Device &rhs);

    TS_DEBUG_API bool operator<(const Device &lhs, const Device &rhs);

    TS_DEBUG_API bool operator>(const Device &lhs, const Device &rhs);

    TS_DEBUG_API bool operator<=(const Device &lhs, const Device &rhs);

    TS_DEBUG_API bool operator>=(const Device &lhs, const Device &rhs);

    class TS_DEBUG_API DeviceMismatchException : public Exception {
    public:
        using self = DeviceMismatchException;
        using supper = Exception;

        explicit DeviceMismatchException(const Device &needed, const Device &given);

        static std::string DeviceMismatchMessage(const Device &needed, const Device &given);

        const Device &needed() const { return m_needed; }

        const Device &given() const { return m_given; }

    private:
        Device m_needed;
        Device m_given;

    public:
        DeviceMismatchException(const self &other) = default;

        DeviceMismatchException &operator=(const self &other) = default;

        DeviceMismatchException(self &&other) {
            *this = std::move(other);
        }

        DeviceMismatchException &operator=(self &&other) TS_NOEXCEPT {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
            MOVE_MEMBER(m_needed);
            MOVE_MEMBER(m_given);
#undef MOVE_MEMBER
            return *this;
        }
    };

    class TS_DEBUG_API MemoryDevice : public Device {
    public:
        using self = MemoryDevice;
        using supper = Device;

        MemoryDevice(const DeviceType &type, int id) : supper(type, id) {}

        MemoryDevice(const DeviceType &type) : supper(type) {}
        
        MemoryDevice(const char *type) : supper(type) {}

        MemoryDevice() : supper() {}

        static const self &portal() {
            static self _portal(PORTAL, PORTAL_ID);
            return _portal;
        }

        MemoryDevice(const self &other) = default;

        MemoryDevice &operator=(const self &other) = default;

        MemoryDevice(self &&other) {
            *this = std::move(other);
        }

        MemoryDevice &operator=(self &&other) {
            supper::operator=(std::move(other));
            return *this;
        }

    };

    class TS_DEBUG_API ComputingDevice : public Device {
    public:
        using self = ComputingDevice;
        using supper = Device;

        ComputingDevice(const DeviceType &type, int id) : supper(type, id) {}

        ComputingDevice(const DeviceType &type) : supper(type) {}

        ComputingDevice(const char *type) : supper(type) {}

        ComputingDevice() : supper() {}

        static const self &portal() {
            static self _portal(PORTAL, PORTAL_ID);
            return _portal;
        }

        ComputingDevice(const self &other) = default;

        ComputingDevice &operator=(const self &other) = default;

        ComputingDevice(self &&other) {
            *this = std::move(other);
        }

        ComputingDevice &operator=(self &&other) {
            supper::operator=(std::move(other));
            return *this;
        }
    };
}

namespace std {
    template<>
    struct hash<ts::Device> {
        std::size_t operator()(const ts::Device &key) const {
            using std::size_t;
            using std::hash;

            return hash<ts::DeviceType>()(key.type())
                   ^ hash<int>()(key.id());
        }
    };

    template<>
    struct hash<ts::MemoryDevice> {
        using type = ts::MemoryDevice;
        std::size_t operator()(const type &key) const {
            return hash<type::supper>()(key);
        }
    };

    template<>
    struct hash<ts::ComputingDevice> {
        using type = ts::ComputingDevice;
        std::size_t operator()(const type &key) const {
            return hash<type::supper>()(key);
        }
    };
}

#endif //TENSORSTACK_CORE_DEVICE_H
