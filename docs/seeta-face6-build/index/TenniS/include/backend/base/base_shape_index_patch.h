//
// Created by kier on 2019/3/14.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_SHAPE_INDEX_PATCH_H
#define TENSORSTACK_BACKEND_BASE_BASE_SHAPE_INDEX_PATCH_H


#include "operator_on_device.h"
#include "backend/common_structure.h"

namespace ts {
    namespace base {
        class ShapeIndexPatch : public OperatorOnDevice {
        public:
            using self = ShapeIndexPatch;
            using supper = OperatorOnDevice;

            static const std::string param_origin_patch;
            static const std::string param_origin;

            ShapeIndexPatch();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param x `[number, channels, height, width]`
             * @param pos `[number, landmark, 1, 1]`
             * @param origin_patch `Int[2]{h, w}`
             * @param origin `Int[2]{h, w}`
             * @param out `[number, channels, x_patch_h, landmark / 2, x_patch_w]`
             *      `x_patch_h = int(origin_patch.h * x.height / origin.h + 0.5)`,
             *      `x_patch_w = int(origin_patch.w * x.width / origin.w + 0.5)`,
             */
            virtual void sample(const Tensor &x, const Tensor &pos, const Size2D &origin_patch, const Size2D &origin, Tensor &out) = 0;

        private:
            Size2D m_origin_patch;
            Size2D m_origin;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_SHAPE_INDEX_PATCH_H
