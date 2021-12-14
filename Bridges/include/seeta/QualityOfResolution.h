//
// Created by kier on 2019-07-24.
//

#ifndef SEETA_QUALITYEVALUATOR_QUALITYOFRESOLUTION_H
#define SEETA_QUALITYEVALUATOR_QUALITYOFRESOLUTION_H

#include "QualityStructure.h"

namespace seeta {
    namespace v3 {
        class QualityOfResolution : public QualityRule {
        public:
            using self = QualityOfResolution;
            using supper = QualityRule;

            /**
             * Construct with recommend parameters
             */
            SEETA_API QualityOfResolution();

            /**
             *
             * @param low
             * @param high
             * [0, low) and [v3, ~) => LOW
             * [low, high) and [v2, v3) => MEDIUM
             * [high, ~) => HIGH
             */
            SEETA_API QualityOfResolution(float low, float high);

            SEETA_API ~QualityOfResolution() override;

            SEETA_API QualityResult check(
                    const SeetaImageData &image,
                    const SeetaRect &face,
                    const SeetaPointF *points,
                    const int32_t N) override;

        private:
            QualityOfResolution(const QualityOfResolution &) = delete;
            QualityOfResolution &operator=(const QualityOfResolution &) = delete;

        private:
            void *m_data;
        };
    }
    using namespace v3;
}

#endif //SEETA_QUALITYEVALUATOR_QUALITYOFRESOLUTION_H
