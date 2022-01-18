#include "compiler/option/translator_option.h"

namespace ts {

    static std::vector<const TranslatorOption*> &GetStaticTranslateOptions() {
        static std::vector<const TranslatorOption*> options;
        return options;
    }

    const std::vector<const TranslatorOption*> &GetFullTranslateOptions() {
        return GetStaticTranslateOptions();
    }

    void RegisterTranslateOption(const TranslatorOption *option) {
        auto &options = GetStaticTranslateOptions();
        options.emplace_back(option);
    }

    static std::vector<const TranslatorV2Option*> &GetStaticTranslateV2Options() {
        static std::vector<const TranslatorV2Option*> options;
        return options;
    }

    const std::vector<const TranslatorV2Option*> &GetFullTranslateV2Options() {
        return GetStaticTranslateV2Options();
    }

    void RegisterTranslateV2Option(const TranslatorV2Option *option) {
        auto &options = GetStaticTranslateV2Options();
        options.emplace_back(option);
    }
}

