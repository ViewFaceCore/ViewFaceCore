//
// Created by kier on 19-4-24.
//

#ifndef SEETA_EyeStateDetector_H
#define SEETA_EyeStateDetector_H

#include "Common/Struct.h"

namespace seeta {
    namespace v6 {
        class EyeStateDetector {
        public:
            enum EYE_STATE {EYE_CLOSE, EYE_OPEN, EYE_RANDOM, EYE_UNKNOWN};

            SEETA_API explicit EyeStateDetector(const seeta::ModelSetting &setting);
            SEETA_API ~EyeStateDetector();


            SEETA_API void  Detect(const SeetaImageData &image, const SeetaPointF *points, EYE_STATE &leftstate, EYE_STATE &rightstate);
           

            enum Property {
                 PROPERTY_NUMBER_THREADS = 4,
                 PROPERTY_ARM_CPU_MODE = 5
            };

            SEETA_API void set(Property property, double value); 

            SEETA_API double get(Property property) const;
 

        private:
            EyeStateDetector(const EyeStateDetector &) = delete;
            const EyeStateDetector &operator=(const EyeStateDetector&) = delete;

        private:
            class Implement;
            Implement *m_impl;
        };
    }
    using namespace v6;
}

#endif
