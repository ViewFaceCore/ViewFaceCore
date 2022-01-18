//
// Created by kier on 19-7-24.
//

#ifndef SEETA_QULITY_EVALUATOR_QULITY_STRUCTURE_H
#define SEETA_QULITY_EVALUATOR_QULITY_STRUCTURE_H

#include "Common/Struct.h"

namespace seeta {
    namespace v3 {
        enum QualityLevel {
            LOW = 0,
            MEDIUM = 1,
            HIGH = 2,
        };

        class QualityResult {
        public:
            using self = QualityResult;

            QualityResult() = default;

            QualityResult(QualityLevel level, float score = 0)
                : level(level), score(score) {}

            QualityLevel level = LOW;   ///< quality level
            float score = 0;            ///< greater means better, no range limit
        };

        struct QualityResultEx {
            int attr;
            QualityLevel level;   ///< quality level
            float score;          ///< greater means better, no range limit
        };

        struct QualityResultExArray {
            int size;
            QualityResultEx *data;
        }; 

        class QualityRule {
        public:
            using self = QualityRule;

            virtual ~QualityRule() = default;

            /**
             *
             * @param image original image
             * @param face face location
             * @param points landmark on face
             * @param N how many landmark on face given, normally 5
             * @return Quality result
             */
            virtual QualityResult check(
                    const SeetaImageData &image,
                    const SeetaRect &face,
                    const SeetaPointF *points,
                    int32_t N) = 0;
        };
    }
    using namespace v3;
}

#endif //SEETA_QULITY_EVALUATOR_QULITY_STRUCTURE_H
