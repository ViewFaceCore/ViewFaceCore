//
// Created by kier on 19-7-31.
//

#ifndef SEETA_FACERECOGNIZER_TRANSFORM_H
#define SEETA_FACERECOGNIZER_TRANSFORM_H

#include "api/cpp/graphics2d.h"
#include <cmath>
#include <vector>

namespace seeta {
    /**
     *
     * @param src length N * 2
     * @param dst length N * 2
     * @param N
     * @param M length 3 * 3
     * @return
     * compute transform matrix which can trans points from src to dst, src = M * dst
     * @note M can directly parse to ts::intime::afffine_sample2d
     */
    bool transform2d(
            const float *src,
            const float *dst,
            int N,
            float *M);

    ts::Trans2D<float> transform2d(const std::vector<ts::Vec2D<float>> &src, const std::vector<ts::Vec2D<float>> &dst);

    class SimilarityTransform2D {
    public:
        SimilarityTransform2D()
                : params(ts::affine::identity<float>()) {}

        SimilarityTransform2D(const ts::Trans2D<float> &matrix)
                : params(matrix) {}

        SimilarityTransform2D(float scale, float rotation = 0, const ts::Vec2D<float> &translation = {0, 0}) {

            this->params = {
                    std::cos(rotation), - std::sin(rotation), 0,
                    std::sin(rotation),   std::cos(rotation), 0,
                                     0,                    0, 1
            };
            params.data(0) *= scale;
            params.data(1) *= scale;
            params.data(3) *= scale;
            params.data(4) *= scale;
            params.data(2) = translation.data(0);
            params.data(5) = translation.data(1);
        }

        static ts::Trans2D<float> _umeyama(const std::vector<ts::Vec2D<float>> &src, const std::vector<ts::Vec2D<float>> &dst);

        bool estimate(const std::vector<ts::Vec2D<float>> &src, const std::vector<ts::Vec2D<float>> &dst) {
            this->params = _umeyama(src, dst);
            return true;
        }

        ts::Trans2D<float> params;
    };
}


#endif //SEETA_FACERECOGNIZER_TRANSFORM_H
