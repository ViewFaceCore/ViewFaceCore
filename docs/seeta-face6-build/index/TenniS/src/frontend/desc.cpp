//
// Created by kier on 2019-04-13.
//

#include "frontend/desc.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace desc {
        Bubble resize2d(ResizeType type) {
            Bubble bubble(name::layer::resize2d(), name::layer::resize2d());
            bubble.set(name::type, tensor::from(int32_t(type)));
            return std::move(bubble);
        }

        Bubble add() {
            return Bubble(name::layer::add(), name::layer::add());
        }

        Bubble sub() {
            return Bubble(name::layer::sub(), name::layer::sub());
        }

        Bubble mul() {
            return Bubble(name::layer::mul(), name::layer::mul());
        }

        Bubble div() {
            return Bubble(name::layer::div(), name::layer::div());
        }

        Bubble transpose(const std::vector<int32_t> &permute) {
            Bubble bubble(name::layer::transpose(), name::layer::transpose());
            bubble.set(name::permute, tensor::build(INT32, permute));
            return std::move(bubble);
        }

        Bubble sigmoid() {
            return Bubble(name::layer::sigmoid(), name::layer::sigmoid());
        }

        Bubble gather(int32_t axis) {
            Bubble bubble(name::layer::gather(), name::layer::gather());
            bubble.set(name::axis, tensor::from<int32_t>(axis));
            return std::move(bubble);
        }

        Bubble concat(int32_t dim) {
            Bubble bubble(name::layer::concat(), name::layer::concat());
            bubble.set(name::dim, tensor::from<int32_t>(dim));
            return std::move(bubble);
        }

        Bubble softmax(int32_t dim, bool smooth) {
            Bubble bubble(name::layer::softmax(), name::layer::softmax());
            bubble.set(name::dim, tensor::from<int32_t>(dim));
            bubble.set(name::smooth, tensor::from<bool>(smooth));
            return std::move(bubble);
        }

        Bubble pad(float padding_value) {
            Bubble bubble(name::layer::pad(), name::layer::pad());
            bubble.set(name::padding_value, tensor::from<float>(padding_value));
            return std::move(bubble);
        }

        Bubble cast(DTYPE dtype) {
            Bubble bubble(name::layer::cast(), name::layer::cast());
            bubble.set(name::dtype, tensor::from<int>(dtype));
            return std::move(bubble);
        }

        Bubble affine_sample2d(int32_t dim, float outer_value, ResizeType type) {
            Bubble bubble(name::layer::affine_sample2d(), name::layer::affine_sample2d());
            bubble.set(name::dim, tensor::from<int32_t>(dim));
            bubble.set(name::outer_value, tensor::from<float>(outer_value));
            bubble.set(name::type, tensor::from(int32_t(type)));
            return std::move(bubble);
        }

        Bubble affine_on_sample2d(int32_t dim, ResizeType type) {
            Bubble bubble(name::layer::affine_sample2d(), name::layer::affine_sample2d());
            bubble.set(name::dim, tensor::from<int32_t>(dim));
            bubble.set(name::type, tensor::from(int32_t(type)));
            return std::move(bubble);
        }

        Bubble matmul(bool transpose) {
            Bubble bubble(name::layer::inner_prod(), name::layer::inner_prod());
            bubble.set("transpose", tensor::from<bool>(transpose));
            return std::move(bubble);
        }

        Bubble broadcast() {
            return Bubble("broadcast", "broadcast");
        }
    }
}
