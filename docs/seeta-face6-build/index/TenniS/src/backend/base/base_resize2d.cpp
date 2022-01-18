//
// Created by kier on 2019/2/20.
//

#include <backend/base/base_resize2d.h>

#include "backend/base/base_resize2d.h"
#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        Resize2D::Resize2D() {
            field(name::type, OPTIONAL, tensor::from(int(Resize2DType::LINEAR)));
        }

        void Resize2D::init() {
            m_type = Resize2DType(tensor::to_int(get(name::type)));
        }

        class _Resize2DTip {
        public:
            int dim;
            int height;
            int width;
        };

        int find_resized_height_dim(const int *shape_data, const int *size_data, int n) {
            auto resized_height_dim = -1;
            for (int i = 0; i < n; ++i) {
                if (size_data[i] == 0) return -1;
                if (size_data[i] < 0) continue;
                if (shape_data[i] == size_data[i]) {
                    bool satisfied_tail = true;
                    for (int j = i + 2; j < n; ++j) {
                        if (shape_data[j] == size_data[j] || size_data[j] < 0) {
                            continue;
                        } else {
                            satisfied_tail = false;
                            break;
                        }
                    }
                    if (!satisfied_tail) continue;
                }
                resized_height_dim = i;
                break;
            }
            if (resized_height_dim < 0 ||
                resized_height_dim >= n - 1 ||
                size_data[resized_height_dim + 1] <= 0) {
                return -1;
            }
            for (int i = resized_height_dim + 2; i < n; ++i) {
                if (size_data[i] == 0) return -1;
                if (shape_data[i] == size_data[i] || size_data[i] < 0) continue;
                return -1;
            }
            return resized_height_dim;
        }

        static Tensor::Prototype check_outputs(const Stack &stack, int &dim, bool &ready) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &x = stack[0];
            auto size = tensor::cast(INT32, stack[1]);

            TS_AUTO_CHECK(size.dims() == 1 && x.dims() == size.size(0));
            TS_AUTO_CHECK(size.size(0) >= 2);

            auto &shape = x.sizes();
            auto shape_data = shape.data();

            auto n = size.size(0);
            auto size_data = size.data<int32_t>();

            auto resized_height_dim = find_resized_height_dim(shape_data, size_data, n);
            if (resized_height_dim < 0) {
                TS_LOG_ERROR << "Can not resize " << to_string(x.sizes())
                             << " to " << to_string(Shape(size_data, size_data + n)) << eject;
            }
            dim = resized_height_dim;

            Shape resized_shape = x.sizes();

            ready = resized_shape[resized_height_dim] == size_data[resized_height_dim] &&
                    resized_shape[resized_height_dim + 1] == size_data[resized_height_dim + 1];

            resized_shape[resized_height_dim] = size_data[resized_height_dim];
            resized_shape[resized_height_dim + 1] = size_data[resized_height_dim + 1];

            return Tensor::Prototype(x.dtype(), resized_shape);
        }

        int Resize2D::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            int dim;
            bool ready;
            auto resized_proto = check_outputs(stack, dim, ready);

            output.resize(1);
            output[0] = resized_proto;

            return 1;
        }

        int Resize2D::run(Stack &stack) {
            int dim;
            bool ready = false;
            auto resized_proto = check_outputs(stack, dim, ready);

            if (ready) {
                stack.push(0);
                return 1;
            }

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto out = *stack.push(resized_proto, memory_device);

            resize2d(x, dim, m_type, out);

            return 1;
        }
    }
}
