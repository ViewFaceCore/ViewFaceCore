//
// Created by kier on 2018/6/29.
//

#ifndef TENSORSTACK_OPERATOR_FACTORY_H
#define TENSORSTACK_OPERATOR_FACTORY_H

#include <functional>
#include "runtime/operator.h"
#include "utils/static.h"

#include <set>
#include <map>

namespace ts {

    class TS_DEBUG_API OperatorNotFoundException : public Exception {
    public:
        explicit OperatorNotFoundException(const DeviceType &device_type, const std::string &operator_name)
                : Exception(OperatorNotFoundMessage(device_type, operator_name))
                , m_device_type(device_type), m_operator_name(operator_name) {
        }

        static std::string OperatorNotFoundMessage(const DeviceType &device_type, const std::string &operator_name) {
            std::ostringstream oss;
            oss << "No operator \"" << operator_name << "\" registered on device \"" << device_type << "\".";
            return oss.str();
        }

        const DeviceType &device_type() const {
            return m_device_type;
        }

        const std::string &operator_name() const {
            return m_operator_name;
        }

    private:
        DeviceType m_device_type;
        std::string m_operator_name;
    };

    class TS_DEBUG_API OperatorCreator {
    public:

        using function =  std::function<Operator::shared(void)>;
        using CreatorFucMap = std::map<std::pair<DeviceType, std::string>, OperatorCreator::function>;

        Operator::shared OperatorCreatorFunction();

        static function Query(const DeviceType &device_type,
                              const std::string &operator_name) TS_NOEXCEPT;

        static void Register(const DeviceType &device_type,
                             const std::string &operator_name,
                             const function &operator_creator) TS_NOEXCEPT;

        static const CreatorFucMap GetCreatorFucMap();

        static void flush(const CreatorFucMap& creator_map);

        static Operator::shared CreateNoException(const DeviceType &device_type,
                                                  const std::string &operator_name,
                                                  bool strict = false) TS_NOEXCEPT;

        /**
         * Will throw OperatorNotFoundException if operator not found
         * @param device_type
         * @param operator_name
         * @param strict
         * @return
         */
        static Operator::shared Create(const DeviceType &device_type,
                                       const std::string &operator_name,
                                       bool strict = false);

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
#define TS_REGISTER_OPERATOR(CLASS_NAME, DEVICE_TYPE, OP_NAME) \
    namespace \
    { \
        static ts::Operator::shared _ts_concat_name(CLASS_NAME, _CREATOR)() { return std::make_shared<CLASS_NAME>(); } \
        ts::StaticAction ts_serial_name(_ts_static_action_)(ts::OperatorCreator::Register, DEVICE_TYPE, OP_NAME, _ts_concat_name(CLASS_NAME, _CREATOR)); \
    }


#endif //TENSORSTACK_OPERATOR_FACTORY_H
