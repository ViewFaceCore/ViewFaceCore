//
// Created by kier on 2019/10/21.
//

#include "backend/base/base_broadcast_v2.h"

#include "utils/assert.h"
#include "runtime/stack.h"

#include <numeric>
#include <core/tensor_builder.h>

namespace ts {
    namespace base {
        static inline void front_append_ones(Shape &shape, int count) {
            Shape ones(count, 1);
            shape.insert(shape.begin(), ones.begin(), ones.end());
        }

        void BroadcastV2::init() {
            supper::init();
        }

        int BroadcastV2::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &x = stack[0];
            auto &shape = stack[1];

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), tensor::array::to_int(shape));

            return 1;
        }

        bool BroadcastV2::is_scalar(const Shape &shape) {
            return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) == 1;
        }

        void BroadcastV2::broad_with_bias(const Tensor &x, Tensor &out, int dim) {
            (void) (dim);
            this->broadcast(x, out);
        }

        void BroadcastV2::broadcast_with_scalar(const Tensor &x, Tensor &out) {
            this->broadcast(x, out);
        }

        static bool reduce_shape(Shape &lhs, Shape &rhs, Shape &out) {
            bool reduced = false;
            bool lhs_ones = false;
            bool rhs_ones = false;
            bool lhs_equal_rhs = false;
            for (size_t i = 0; i < out.size(); ) {
                lhs_ones = lhs[i] == 1;
                rhs_ones = rhs[i] == 1;
                lhs_equal_rhs = lhs[i] == rhs[i];
                if (!(lhs_ones || rhs_ones || lhs_equal_rhs)) {
                    ++i;
                    continue;
                }
                auto j = i + 1;
                for (; j < out.size(); ++j) {
                    if (lhs_ones) lhs_ones = lhs[j] == 1;
                    if (rhs_ones) rhs_ones = rhs[j] == 1;
                    if (lhs_equal_rhs) lhs_equal_rhs = lhs[j] == rhs[j];
                    if (!(lhs_ones || rhs_ones || lhs_equal_rhs)) break;
                }
                if (j - i > 1) {
                    auto a = std::accumulate(lhs.begin() + i, lhs.begin() + j, 1, std::multiplies<int32_t>());
                    auto b = std::accumulate(rhs.begin() + i, rhs.begin() + j, 1, std::multiplies<int32_t>());
                    auto c = std::accumulate(out.begin() + i, out.begin() + j, 1, std::multiplies<int32_t>());

                    lhs.erase(lhs.begin() + i, lhs.begin() + j);
                    rhs.erase(rhs.begin() + i, rhs.begin() + j);
                    out.erase(out.begin() + i, out.begin() + j);

                    lhs.insert(lhs.begin() + i, a);
                    rhs.insert(rhs.begin() + i, b);
                    out.insert(out.begin() + i, c);

                    reduced = true;
                    ++i;
                    continue;
                } else {
                    i = j + 1;
                    continue;
                }
            }
            return reduced;
        }

        int BroadcastV2::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto x = *stack.index(0);
            auto shape = *stack.index(1);

            auto x_shape = x.sizes();
            auto out_shape = Shape(tensor::array::to_int(shape));

            bool do_broadcast = broadcast(x_shape, out_shape);

            if (!do_broadcast) {
                stack.push(x.reshape(out_shape));
                return 1;
            }

            auto out_proto = Tensor::Prototype(x.dtype(), out_shape);

            auto memory_device = running_memory_device();

            x = x.view(memory_device).reshape(x_shape);    // do sync, and set default data to given device
            auto out = *stack.push(out_proto, memory_device);

            {
                auto y_shape = out_shape;
                if (reduce_shape(y_shape, x_shape, out_shape)) {
                    x = x.reshape(x_shape);
                    out = out.reshape(out_shape);
                }
            }

            int dim;
            if (is_scalar(x_shape)) {
                broadcast_with_scalar(x, out);
            } else if (is_bias(out_shape, x_shape, dim)) {
                broad_with_bias(x, out, dim);
            } else {
                broadcast(x, out);
            }

            return 1;
        }

        bool BroadcastV2::is_bias(Shape &lhs_shape, Shape &rhs_shape, int &dim) {
            auto count = std::accumulate(rhs_shape.begin(), rhs_shape.end(), 1, std::multiplies<int>());
            for (size_t i = 0; i < rhs_shape.size(); ++i) {
                if (rhs_shape[i] == count && lhs_shape[i] == count) {
                    dim = int(i);
                    return true;
                }
            }
            return false;
        }

        bool BroadcastV2::broadcast(Shape &x, const Shape &shape) {
            if (x.size() > shape.size()) {
                TS_LOG_ERROR << "Can not broadcast " << to_string(x) << " to " << to_string(shape) << eject;
            }
            if (x.size() < shape.size()) {
                front_append_ones(x, int(shape.size() - x.size()));
            }
            if (x == shape) return false;
            auto N = x.size();
            for (size_t i = 0; i < N; ++i) {
                if (x[i] != shape[i] && x[i] != 1) {
                    TS_LOG_ERROR << "Can not broadcast " << to_string(x) << " to " << to_string(shape) << eject;
                }
            }
            return true;
        }
    }
}
