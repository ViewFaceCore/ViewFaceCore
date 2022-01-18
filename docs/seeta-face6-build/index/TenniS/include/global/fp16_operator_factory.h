#ifndef TENSORSTACK_GLOBAL_FP16_OPERATOR_FACTORY_H
#define TENSORSTACK_GLOBAL_FP16_OPERATOR_FACTORY_H

#include <functional>
#include "runtime/operator.h"
#include "utils/static.h"

#include <set>

namespace ts {

    class TS_DEBUG_API Fp16OperatorCreator {
    public:

        using function = std::function<Operator::shared(void)>;

        static function Query(const DeviceType &device_type,
            const std::string &operator_name) TS_NOEXCEPT;

        static void Register(const DeviceType &device_type,
            const std::string &operator_name,
            const function &operator_creator) TS_NOEXCEPT;

        static Operator::shared Create(const DeviceType &device_type,
            const std::string &operator_name,
            bool strict = false) TS_NOEXCEPT;

        static function Query(const DeviceType &device_type,
            const std::string &operator_name, bool strict) TS_NOEXCEPT;

        static void Clear();

        /**
         * @return set of pair of device type and operator name
         */
        static std::set<std::pair<std::string, std::string>> AllKeys() TS_NOEXCEPT;
    };
}

/**
* Static action
*/
#define TS_REGISTER_FP16_OPERATOR(CLASS_NAME, DEVICE_TYPE, OP_NAME) \
    namespace \
    { \
        static ts::Operator::shared _ts_concat_name(CLASS_NAME, _FP16_CREATOR)() { return std::make_shared<CLASS_NAME>(); } \
        ts::StaticAction ts_serial_name(_ts_static_action_)(ts::Fp16OperatorCreator::Register, DEVICE_TYPE, OP_NAME, _ts_concat_name(CLASS_NAME, _FP16_CREATOR)); \
    }


#endif //TENSORSTACK_GLOBAL_FP16_OPERATOR_FACTORY_H
