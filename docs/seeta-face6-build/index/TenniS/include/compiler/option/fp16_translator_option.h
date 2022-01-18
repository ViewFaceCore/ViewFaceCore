//
// Created by kier on 2019-06-10.
//

#ifndef TENSORSTACK_FP16_TRANSLATOR_OPTION_H
#define TENSORSTACK_FP16_TRANSLATOR_OPTION_H

#include "translator_option.h"

namespace ts {

    class Fp16TranslatorOption : public TranslatorOption {
    public:
        bool translate(const ComputingDevice &device,
                       const Node node,
                       Node &translated_node,
                       const std::string &params,
                       bool output_flag) const final;
    };

}


#endif //TENSORSTACK_FP16_TRANSLATOR_OPTION_H
