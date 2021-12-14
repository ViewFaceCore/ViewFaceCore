//
// Created by kier on 19-4-24.
//

#ifndef SEETA_FACERECOGNIZER_FACERECOGNIZER_H
#define SEETA_FACERECOGNIZER_FACERECOGNIZER_H

#include "Common/Struct.h"
#include "seeta/SeetaFaceRecognizerConfig.h"

// #define SEETA_FACE_RECOGNIZER_MAJOR_VERSION 6
// #define SEETA_FACE_RECOGNIZER_MINOR_VERSION 0
// #define SEETA_FACE_RECOGNIZER_SINOR_VERSION 0

namespace seeta {
    namespace SEETA_FACE_RECOGNIZE_NAMESPACE_VERSION {
        class FaceRecognizer {
        public:
            using self = FaceRecognizer;
            enum Property {
                 PROPERTY_NUMBER_THREADS = 4,
                 PROPERTY_ARM_CPU_MODE = 5
            };

            SEETA_API explicit FaceRecognizer(const SeetaModelSetting &setting);
            SEETA_API ~FaceRecognizer();

            SEETA_API FaceRecognizer(const self *other);

            SEETA_API static int GetCropFaceWidth();
            SEETA_API static int GetCropFaceHeight();
            SEETA_API static int GetCropFaceChannels();

            SEETA_API static bool CropFace(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face);

            SEETA_API int GetExtractFeatureSize() const;

            SEETA_API bool ExtractCroppedFace(const SeetaImageData &image, float *features) const;

            SEETA_API bool Extract(const SeetaImageData &image, const SeetaPointF *points, float *features) const;

            SEETA_API float CalculateSimilarity(const float *features1, const float *features2) const;

            static seeta::ImageData CropFace(const SeetaImageData &image, const SeetaPointF *points) {
                seeta::ImageData face(GetCropFaceWidth(), GetCropFaceHeight(), GetCropFaceChannels());
                CropFace(image, points, face);
                return face;
            }

            SEETA_API int GetCropFaceWidthV2() const;

            SEETA_API int GetCropFaceHeightV2() const;

            SEETA_API int GetCropFaceChannelsV2() const;

            SEETA_API bool CropFaceV2(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face);

            seeta::ImageData CropFaceV2(const SeetaImageData &image, const SeetaPointF *points) {
                seeta::ImageData face(GetCropFaceWidthV2(), GetCropFaceHeightV2(), GetCropFaceChannelsV2());
                CropFaceV2(image, points, face);
                return face;
            }

            SEETA_API void set(Property property, double value);

            SEETA_API double get(Property property) const;


        private:
            FaceRecognizer(const FaceRecognizer &) = delete;
            const FaceRecognizer &operator=(const FaceRecognizer&) = delete;

        private:
            class Implement;
            Implement *m_impl;
        };
    }
    using namespace SEETA_FACE_RECOGNIZE_NAMESPACE_VERSION;
}

#endif //SEETA_FACERECOGNIZER_FACERECOGNIZER_H
