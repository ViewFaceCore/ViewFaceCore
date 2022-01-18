#include <kernels/cpu/pad.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

#include <core/memory.h>
#include <numeric>

namespace ts {
    namespace cpu {
        namespace {
            class copy_menu {
            public:
                int dst_index;
                int src_index;
                int count;
            };

            class set_menu {
            public:
                int dst_index;
                int count;
            };

            class pad_menu {
            public:
                set_menu head;
                copy_menu body;
                set_menu tail;
            };
        }

        static inline pad_menu get_pad_menu(int size, const std::array<int, 2> &padding) {
            int head = -padding[0];
            int tail = size + padding[1];

            if (head >= tail) {
                return {{0, 0},
                        {0, 0, 0},
                        {0, 0}};
            }

            int head_count = -head;
            set_menu menu_head = head_count > 0
                                 ? set_menu({0, head_count})
                                 : set_menu({0, 0});

            int body_left = std::max<int>(head, 0);
            int body_right = std::min<int>(tail, size);
            int body_count = body_right - body_left;
            copy_menu menu_body = body_count > 0
                                  ? copy_menu({body_left - head, body_left, body_count})
                                  : copy_menu({body_left - head, body_left, 0});

            int tail_count = tail - size;
            set_menu menu_tail = tail_count > 0
                                 ? set_menu({size - head, tail_count})
                                 : set_menu({size - head, 0});

            return {menu_head, menu_body, menu_tail};
        }

        static void pad2d(const Tensor &x, int dim,
                const std::array<int, 2> &padding_h, const std::array<int, 2> &padding_w, float padding_value, Tensor &out) {

            auto &shape = x.sizes();
            auto number = std::accumulate(shape.begin(), shape.begin() + dim, 1, std::multiplies<int>());
            auto width = std::accumulate(shape.begin() + dim + 2, shape.end(), 1, std::multiplies<int>());

            auto bytes = x.proto().type_bytes();
            auto device_id = out.device().id();

            Tensor cpu_padding_data;
            if (tensor::support(x.dtype())) {
                cpu_padding_data = tensor::build(x.dtype(), padding_value);
            } else {
                cpu_padding_data = Tensor(x.dtype(), {1});
                std::memset(cpu_padding_data.data(), 0, size_t(bytes));
            }
            Tensor padding_data = cpu_padding_data.view(out.device());

            auto menu_h = get_pad_menu(shape[dim], padding_h);
            auto menu_w = get_pad_menu(shape[dim + 1], padding_w);

            auto memcpy_handler = HardConverter::Query(out.device().type(), x.device().type());
            TS_AUTO_CHECK(memcpy_handler != nullptr);

            HypeShape src_shape({number, x.size(dim), x.size(dim + 1), width});
            HypeShape dst_shape({number, out.size(dim), out.size(dim + 1), width});

            for (int n = 0; n < number; ++n) {
                // memset top
                if (menu_h.head.count) {
                    auto dst_index = dst_shape.to_index({n, menu_h.head.dst_index, 0, 0}) * bytes;
                    auto set_count = menu_h.head.count * dst_shape.shape(2) * dst_shape.shape(3) * bytes;
                    memset(out.data<char>() + dst_index, out.device(), set_count,
                           padding_data.data(), padding_data.device(), bytes);
                }
                // memset left side
                if (menu_w.head.count) {
                    for (int h = 0; h < menu_h.body.count; ++h) {
                        auto dst_index =
                                dst_shape.to_index({n, menu_h.body.dst_index + h, menu_w.head.dst_index, 0}) * bytes;
                        auto set_count = menu_w.head.count * dst_shape.shape(3) * bytes;
                        memset(out.data<char>() + dst_index, out.device(), set_count,
                               padding_data.data(), padding_data.device(), bytes);
                    }
                }
                // memset right side
                if (menu_w.tail.count) {
                    for (int h = 0; h < menu_h.body.count; ++h) {
                        auto dst_index =
                                dst_shape.to_index({n, menu_h.body.dst_index + h, menu_w.tail.dst_index, 0}) * bytes;
                        auto set_count = menu_w.tail.count * dst_shape.shape(3) * bytes;
                        memset(out.data<char>() + dst_index, out.device(), set_count,
                               padding_data.data(), padding_data.device(), bytes);
                    }
                }
                // memset bottom
                if (menu_h.tail.count) {
                    auto dst_index = dst_shape.to_index({n, menu_h.tail.dst_index, 0, 0}) * bytes;
                    auto set_count = menu_h.tail.count * dst_shape.shape(2) * dst_shape.shape(3) * bytes;
                    memset(out.data<char>() + dst_index, out.device(), set_count,
                           padding_data.data(), padding_data.device(), bytes);
                }
                // memcpy
                if (menu_h.body.count && menu_w.body.count) {
                    for (int h = 0; h < menu_h.body.count; ++h) {
                        auto dst_index =
                                dst_shape.to_index({n, menu_h.body.dst_index + h, menu_w.body.dst_index, 0}) * bytes;
                        auto src_index =
                                src_shape.to_index({n, menu_h.body.src_index + h, menu_w.body.src_index, 0}) * bytes;
                        auto copy_count = menu_w.body.count * dst_shape.shape(3) * bytes;
                        memcpy_handler(device_id, out.data<char>() + dst_index,
                                       device_id, x.data<char>() + src_index, copy_count);
                    }
                }
            }
        }

        static void pad1d(const Tensor &x, int dim,
                          const std::array<int, 2> &padding, float padding_value, Tensor &out) {
            if (dim > 0) {
                pad2d(x, dim - 1, {0, 0}, padding, padding_value, out);
            } else if (x.dims() > 1) {
                pad2d(x, dim - 1, {0, 0}, padding, padding_value, out);
            } else {
                auto x_shape = x.sizes();
                x_shape.insert(x_shape.begin(), 1);
                auto out_shape = out.sizes();
                out_shape.insert(out_shape.begin(), 1);
                auto extend_x = x.reshape(x_shape);
                auto extend_out = out.reshape(out_shape);
                pad2d(extend_x, 0, {0, 0}, padding, padding_value, extend_out);
            }
        }

        void Pad::pad(const Tensor &x, const std::vector<std::array<int, 2>> &padding, float padding_value, Tensor &out) {
            int left = 0;
            while (left < padding.size()) {
                if (padding[left][0] != 0 || padding[left][1] != 0) break;
                ++left;
            }
            if (left >= padding.size()) {
                memcpy(out.data(), out.device(), size_t(out.count() * out.proto().type_bytes()),
                        x.data(), x.device(), size_t(x.count() * x.proto().type_bytes()));
                return;
            }

            int right = int(padding.size()) - 1;
            while (right > left) {
                if (padding[right][0] != 0 || padding[right][1] != 0) break;
                --right;
            }

            if (left == right) {
                pad1d(x, left, padding[left], padding_value, out);
                return;
            }

            if (right - left == 1) {
                pad2d(x, left, padding[left], padding[right], padding_value, out);
                return;
            }

            TS_LOG_ERROR << "This version only support 2D or 1D padding" << eject;
        }
    }
}

//using namespace ts;
//using namespace cpu;
//TS_REGISTER_OPERATOR(Pad, CPU, name::layer::pad())
