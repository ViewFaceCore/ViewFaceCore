//
// Created by kier on 2019-07-24.
//

#ifndef SEETA_QUALITYEVALUATOR_QUALITYOFBRIGHTNESS_H
#define SEETA_QUALITYEVALUATOR_QUALITYOFBRIGHTNESS_H

#include "QualityStructure.h"

namespace seeta {
    namespace v3 {
        class QualityOfBrightness : public QualityRule {
        public:
            using self = QualityOfBrightness;
            using supper = QualityRule;

            /**
             * Construct with recommend parameters
             */
            SEETA_API QualityOfBrightness();
            /**
             *
             * @param v0
             * @param v1
             * @param v2
             * @param v3
             * [0, v0) and [v3, ~) => LOW
             * [v0, v1) and [v2, v3) => MEDIUM
             * [v1, v2) => HIGH
             */
            SEETA_API QualityOfBrightness(float v0, float v1, float v2, float v3);

            SEETA_API ~QualityOfBrightness() override;

            SEETA_API QualityResult check(
                    const SeetaImageData &image,
                    const SeetaRect &face,
                    const SeetaPointF *points,
                    const int32_t N) override;

        private:
            QualityOfBrightness(const QualityOfBrightness &) = delete;
            QualityOfBrightness &operator=(const QualityOfBrightness &) = delete;

        private:
            void *m_data;
        };
    }
    using namespace v3;
}

#endif //SEETA_QUALITYEVALUATOR_QUALITYOFBRIGHTNESS_H
