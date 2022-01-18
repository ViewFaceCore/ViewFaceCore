#pragma once

#include "Common/Struct.h"
#include <string>
#include <vector>

namespace seeta
{
    namespace v2
    {
        class MaskDetector {
        public:
            /**
             * \brief initialize `MaskDetector`
             * \param setting one specifc model, or zero model
             */
            SEETA_API explicit MaskDetector(const seeta::ModelSetting &setting = seeta::ModelSetting() );

            SEETA_API ~MaskDetector();

            /**
             * detect if face with mask
             * @param image original image
             * @param face position of face
             * @param score mask confidence
             * @return true for with mask (score >= 0.5)
             */
            SEETA_API bool detect(const SeetaImageData &image, const SeetaRect &face, float *score = nullptr);

        private:
            MaskDetector(const MaskDetector &other ) = delete;
            const MaskDetector &operator=(const MaskDetector &other ) = delete;

        private:
            class Implement;
            Implement *m_impl;
        };
    }
    using namespace v2;
}

