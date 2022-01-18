#ifndef TENSORSTACK_COMPILER_OPTION_TRANSLATOR_OPTION_H
#define TENSORSTACK_COMPILER_OPTION_TRANSLATOR_OPTION_H

#include "module/graph.h"
#include "utils/static.h"
#include "module/module.h"

namespace ts {

    class TranslatorOption {
    public:
        using self = TranslatorOption;

        virtual ~TranslatorOption() = default;

        /**
         * translate node on device
         * @param device computing device
         * @param node wait to translate
         * @param translated node
         * @param output_flag true when current node is output,otherwise false
         * @return translated return true, or false
         */

        virtual bool translate(const ComputingDevice &device,
            const Node node,
            Node &translated_node,
            const std::string &params,
            bool output_flag = false) const = 0;
    };


    class TranslatorV2Option {
    public:
        using self = TranslatorV2Option;

        virtual ~TranslatorV2Option() = default;

        /**
         * translate node on device
         * @param device computing device
         * @param module wait to translate
         */

        virtual Module::shared translate(const ComputingDevice &device,
                                         Module::shared module) const = 0;
    };

    const std::vector<const TranslatorOption*> &GetFullTranslateOptions();

    void RegisterTranslateOption(const TranslatorOption *option);

    const std::vector<const TranslatorV2Option*> &GetFullTranslateV2Options();

    void RegisterTranslateV2Option(const TranslatorV2Option *option);
}

#define TS_REGISTER_TRANSLATOR_OPTION(option) \
namespace { \
    static option ts_serial_name(_ts_translator_option); \
    TS_STATIC_ACTION(ts::RegisterTranslateOption, &(ts_serial_name(_ts_translator_option))); \
}

#define TS_REGISTER_TRANSLATOR_V2_OPTION(option) \
namespace { \
    static option ts_serial_name(_ts_translator_option); \
    TS_STATIC_ACTION(ts::RegisterTranslateV2Option, &(ts_serial_name(_ts_translator_option))); \
}

#endif //TENSORSTACK_COMPILER_OPTION_TRANSLATOR_OPTION_H
