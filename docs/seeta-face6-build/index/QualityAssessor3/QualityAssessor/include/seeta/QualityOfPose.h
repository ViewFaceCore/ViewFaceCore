//
// Created by kier on 2019-07-24.
//

#ifndef SEETA_QUALITYEVALUATOR_QUALITYOFPOSE_H
#define SEETA_QUALITYEVALUATOR_QUALITYOFPOSE_H

#include "QualityStructure.h"

namespace seeta {
    namespace v3 {
        class QualityOfPose : public QualityRule {
        public:
            using self = QualityOfPose;
            using supper = QualityRule;

            /**
             * Construct with recommend parameters
             */
            SEETA_API QualityOfPose();

            SEETA_API ~QualityOfPose() override;

            SEETA_API QualityResult check(
                    const SeetaImageData &image,
                    const SeetaRect &face,
                    const SeetaPointF *points,
                    const int32_t N) override;

        private:
            QualityOfPose(const QualityOfPose &) = delete;
            QualityOfPose &operator=(const QualityOfPose &) = delete;

        private:
            void *m_data;
        };
    }
    using namespace v3;
}

#endif //SEETA_QUALITYEVALUATOR_QUALITYOFPOSE_H
