#ifndef SEETA_AGE_PREDICTOR_H
#define SEETA_AGE_PREDICTOR_H

#include "Common/Struct.h"

#define SEETA_AGE_PREDICTOR_MAJOR_VERSION 6
#define SEETA_AGE_PREDICTOR_MINOR_VERSION 0
#define SEETA_AGE_PREDICTOR_SINOR_VERSION 0

namespace seeta {
    namespace v6 {
        class AgePredictor {
        public:

        enum Property {
                 PROPERTY_NUMBER_THREADS = 4,
                 PROPERTY_ARM_CPU_MODE = 5
            };

            SEETA_API explicit AgePredictor(const SeetaModelSetting &setting);
            SEETA_API ~AgePredictor();

            SEETA_API  int GetCropFaceWidth() const;
            SEETA_API  int GetCropFaceHeight() const;
            SEETA_API  int GetCropFaceChannels() const;
            SEETA_API  bool CropFace(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face) const;

            SEETA_API bool PredictAge(const SeetaImageData &image, int &age) const;

			SEETA_API bool PredictAgeWithCrop(const SeetaImageData &image, const SeetaPointF *points, int &age) const;

            SEETA_API void set(Property property, double value);

            SEETA_API double get(Property property) const;

        private:
            AgePredictor(const AgePredictor &) = delete;
            const AgePredictor &operator=(const AgePredictor&) = delete;

        private:
            class Implement;
            Implement *m_impl;
        };
    }
    using namespace v6;
}

#endif //SEETA_AGE_PREDICTOR_H
