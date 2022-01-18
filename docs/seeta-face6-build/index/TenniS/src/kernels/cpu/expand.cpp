//
// Created by kier on 19-6-17.
//

#include "runtime/operator.h"
#include "runtime/stack.h"
#include "global/operator_factory.h"
#include "core/tensor_builder.h"
#include <algorithm>

#include <algorithm>

namespace ts {
    namespace cpu {
        class Expand : public Operator {
        private:
            int m_front = -1;
            int m_end = -1;
            bool m_inverse = false;

        public:
            Expand() {
                field("front", OPTIONAL, tensor::from<int32_t>(-1));
                field("end", OPTIONAL, tensor::from<int32_t>(-1));
                field("inverse", OPTIONAL, tensor::from<bool>(false));
            }

            void init() final {
                m_front = tensor::to_int(get("front"));
                m_end = tensor::to_int(get("end"));
                m_inverse = tensor::to_bool(get("inverse"));
            }

            static void insert_front(Shape &shape, size_t N) {
                if (N == 0) return;
                Shape ones(N, 1);
                shape.insert(shape.begin(), ones.begin(), ones.end());
            }

            static void insert_end(Shape &shape, size_t N) {
                if (N == 0) return;
                Shape ones(N, 1);
                shape.insert(shape.end(), ones.begin(), ones.end());
            }

            static bool expand_front(Shape &shape, size_t dims, int limit) {
                if (limit < 0) {
                    const auto N = dims - shape.size();
                    insert_front(shape, N);
                    return true;
                } else {
                    const auto N = std::min<size_t>(dims- shape.size(), size_t(limit));
                    insert_front(shape, N);
                    return shape.size() == dims;
                }
            }

            static bool expand_end(Shape &shape, size_t dims, int limit) {
                if (limit < 0) {
                    const auto N = dims - shape.size();
                    insert_end(shape, N);
                    return true;
                } else {
                    const auto N = std::min<size_t>(dims- shape.size(), size_t(limit));
                    insert_end(shape, N);
                    return shape.size() == dims;
                }
            }

            // return true, if satisfied
            bool expand(Shape &shape, size_t dims) {
                if (shape.size() > dims) return false;
                if (shape.size() == dims) return true;
                if (m_front > 0 && m_end > 0 && (shape.size() + size_t(m_front) + size_t(m_end)) < dims) {
                    return false;
                }
                if (m_inverse) {
                    return expand_end(shape, dims, m_end) || expand_front(shape, dims, m_front);
                } else {
                    return expand_front(shape, dims, m_front) || expand_end(shape, dims, m_end);
                }
            }

            std::vector<int32_t> expand(Stack &stack) {
                TS_AUTO_CHECK(stack.size() == 2);

                auto &x = stack[0];
                auto dims = tensor::to_int(stack[1]);

                auto shape = x.sizes();
                if (dims < 0 || !expand(shape, size_t(dims))) {
                    TS_LOG_ERROR << "Can not exapnd " << to_string(x.sizes()) << " to dims=" << dims
                                 << ", with front=" << m_front
                                 << ", end=" << m_end
                                 << std::boolalpha << ", inverse=" << m_inverse
                                 << "." << eject;
                }

                return shape.std();
            }

            int run(Stack &stack) final {
                auto shape = expand(stack);
                auto &x = stack[0];

                stack.push(x.reshape(shape));

                return 1;
            }

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) final {
                auto shape = expand(stack);
                auto &x = stack[0];

                output.resize(1);
                output[0] = Tensor::Prototype(x.dtype(), shape);
                return 1;
            }
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Expand, CPU, "_expand")