#ifndef TENSORSTACK_COMPILER_OPTION_PACK_TRANSLATOR_OPTION_H
#define TENSORSTACK_COMPILER_OPTION_PACK_TRANSLATOR_OPTION_H

#include "translator_option.h"

namespace ts {
    
    class PackTranslatorOption : public TranslatorOption {
    public:
        bool translate(const ComputingDevice &device,
            const Node node,
            Node &translated_node,
            const std::string &params,
            bool output_flag) const final;
    };
}

#endif//TENSORSTACK_COMPILER_OPTION_PACK_TRANSLATOR_OPTION_H