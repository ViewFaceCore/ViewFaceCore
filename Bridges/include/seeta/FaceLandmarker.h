//
// Created by kier on 19-7-16.
//

#ifndef SEETA_FACELANDMARKER_FACELANDMARKER_H
#define SEETA_FACELANDMARKER_FACELANDMARKER_H

#include "Common/Struct.h"
#include "seeta/SeetaFaceLandmarkerConfig.h"

namespace seeta {
    namespace SEETA_FACE_LANDMARKER_NAMESPACE_VERSION {
        class FaceLandmarker {
        public:
            using self = FaceLandmarker;

            SEETA_API explicit FaceLandmarker(const SeetaModelSetting &setting);
            SEETA_API ~FaceLandmarker();

            SEETA_API FaceLandmarker(const self *other);

            SEETA_API int number() const;

            SEETA_API void mark(const SeetaImageData &image, const SeetaRect &face, SeetaPointF *points) const;

            SEETA_API void mark(const SeetaImageData &image, const SeetaRect &face, SeetaPointF *points, int32_t *mask) const;

            std::vector<SeetaPointF> mark(const SeetaImageData &image, const SeetaRect &face) const {
                std::vector<SeetaPointF> points(this->number());
                mark(image, face, points.data());
                return points;
            }

            class PointWithMask {
            public:
                SeetaPointF point;
                bool mask;
            };

            std::vector<PointWithMask> mark_v2(const SeetaImageData &image, const SeetaRect &face) const {
                std::vector<SeetaPointF> points(this->number());
                std::vector<int32_t> masks(this->number());
                mark(image, face, points.data(), masks.data());
                std::vector<PointWithMask> point_with_masks(this->number());
                for (int i = 0; i < this->number(); ++i) {
                    point_with_masks[i].point = points[i];
                    point_with_masks[i].mask = masks[i];
                }
                return point_with_masks;
            }

        private:
            FaceLandmarker(const FaceLandmarker &) = delete;
            const FaceLandmarker &operator=(const FaceLandmarker&) = delete;

        private:
            class Implement;
            Implement *m_impl;
        };
    }
    using namespace SEETA_FACE_LANDMARKER_NAMESPACE_VERSION;
}

#endif //SEETA_FACELANDMARKER_FACELANDMARKER_H
