//
// Created by kier on 2019/4/8.
//

#include <backend/base/base_slice_v3.h>

#include "backend/name.h"
#include "core/tensor_builder.h"

#include <algorithm>
#include <numeric>
#include <cstdint>

#include <climits>

namespace ts {
    namespace base {
        SliceV3::SliceV3() {
        }

        void SliceV3::init() {
            supper::init();
        }

        static int infer_output(int x, int &begin, int &end, int stride, bool begin_flag, bool end_flag) {
            // begin \in [0, x)
            // end \in [-1, x]
            if (begin_flag) {
                begin = stride > 0 ? 0 : x - 1;
            } else {
                if (stride > 0) {
                    if (begin >= x) return 0;  // no elements
                    else if (begin < -x) begin = 0;
                    else if (begin < 0) begin += x;
                } else {
                    if (begin < -x) return 0;  // no elements
                    else if (begin >= x) begin = x - 1;
                    else if (begin < 0) begin += x;
                }
            }
            if (end_flag) {
                end = stride > 0 ? x : -1;
            } else {
                if (stride > 0) {
                    if (end <= -x) return 0;     // no elements
                    else if (end > x) end = x;
                    else if (end < 0) end += x;
                } else {
                    if (end > x) return 0;     // no elements
                    else if (end <= -x) end = -1;
                    else if (end < 0) end += x;
                }
            }

            if (stride > 0) {
                return begin < end ? (end - begin - 1) / stride + 1 : 0;
            } else if (stride < 0) {
                return begin > end ? (begin - end - 1) / -stride + 1 : 0;
            } else {
                TS_LOG_ERROR << "slice step cant not be zero";
                return 0;
            }
        }

        static int bit_ones(uint32_t n) {
            int count = 0;
            while (n != 0) {
                count++;
                n = (n - 1) & n;
            }
            return count;
        }

        static int bit_ones(int32_t n) {
            return bit_ones(uint32_t(n));
        }

        /**
         *
         * @param x [in] raw input shape
         * @param y [out] raw output shape
         * @param begin [in/out]
         * @param end [in/out]
         * @param stride [in/out]
         * @param begin_mask [in]
         * @param end_mask [in]
         * @param ellipsis_mask [in]
         * @param new_axis_mask [in]
         * @param shrink_axis_mask [in]
         * @param in [out] reshape x to in's shape, than do slice by [begin, end, stride]
         * @param out [out] reshape sliced tensor to out's shape
         * @return if infer succeed
         */
        static bool infer_output(
                const Shape &x,
                Shape &y,
                Shape &begin,
                Shape &end,
                Shape &stride,
                int begin_mask,
                int end_mask,
                int ellipsis_mask,
                int new_axis_mask,
                int shrink_axis_mask,
                Shape &in,
                Shape &out) {
            if (stride.empty()) stride.resize(begin.size(), 1);
            if (begin.size() != end.size() || begin.size() != stride.size()) return false;

            auto slice_size = begin.size();
            if (slice_size == 0) return false;

            auto ellipsis_ones = bit_ones(ellipsis_mask);
            if (ellipsis_ones > 1) return false;

            size_t ellipsis_index = 0;
            Shape ellipsis_shape;

            // deal ellipsis
            if (ellipsis_ones) {
                // first deal with in shape
                int left_count = 0;
                int right_count = 0;
                size_t i = 0;
                for (; i < slice_size; ++i) {
                    if (ellipsis_mask & (1 << i)) break;
                    ++left_count;
                    if (new_axis_mask & (1 << i)) --left_count;
                }
                ellipsis_index = i;
                for (++i; i < slice_size; ++i) {
                    ++right_count;
                    if (new_axis_mask & (1 << i)) --right_count;
                }
                auto ellipsis_count = int(x.size()) - left_count - right_count;
                in.clear();
                in.insert(in.end(), x.begin(), x.begin() + left_count);
                in.insert(in.end(), std::accumulate(x.begin() + left_count, x.begin() + left_count + ellipsis_count, 1,
                                                    std::multiplies<int>()));
                in.insert(in.end(), x.begin() + left_count + ellipsis_count, x.end());
                stride[ellipsis_index] = 1;
                begin_mask |= 1 << ellipsis_index;
                end_mask |= 1 << ellipsis_index;

                ellipsis_shape = Shape(x.begin() + left_count, x.begin() + left_count + ellipsis_count);
            } else {
                in = x;
            }

            // deal new_axis
            {
                for (size_t i = 0; i < slice_size; ++i) {
                    if (new_axis_mask & (1 << i)) {
                        if (i > in.size()) return false;
                        in.insert(in.begin() + i, 1);
                    }
                }
            }

            Shape final_shape;

            if (slice_size < in.size()) {
                final_shape = Shape(in.begin() + slice_size, in.end());
                auto final_size = std::accumulate(in.begin() + slice_size, in.end(), 1, std::multiplies<int>());
                in.erase(in.begin() + slice_size, in.end());
                in.push_back(final_size);

                begin.push_back(0);
                end.push_back(0);
                stride.push_back(1);
                begin_mask |= 1 << slice_size;
                end_mask |= 1 << slice_size;

                ++slice_size;
            }

            if (in.size() != slice_size) return false;

            // calculate out
            out.resize(slice_size, 0);
            {
                for (size_t i = 0; i < slice_size; ++i) {
                    out[i] = infer_output(in[i], begin[i], end[i], stride[i], bool(begin_mask & (1 << i)),
                                          bool(end_mask & (1 << i)));
                }
            }
            y = out;

            // deal with may final shape
            {
                if (!final_shape.empty()) {
                    out.pop_back();
                    out.insert(out.end(), final_shape.begin(), final_shape.end());
                    --slice_size;
                }
            }

            // shrink output, and expand ellipsis_shape
            {
                for (int i = int(slice_size) - 1; i >= 0; --i) {
                    if (!ellipsis_shape.empty() && (i == ellipsis_index)) {
                        // expand ellipsis_index
                        out.erase(out.begin() + i);
                        out.insert(out.begin() + i, ellipsis_shape.begin(), ellipsis_shape.end());
                        continue;
                    }
                    if (shrink_axis_mask & (1 << i)) {
                        if (out[i] != 1) return false;
                        out.erase(out.begin() + i);
                    }
                }
            }

            return true;
        }

        static std::string slice_string(
                const Shape &begin,
                const Shape &end,
                const Shape &stride,
                int begin_mask,
                int end_mask,
                int ellipsis_mask,
                int new_axis_mask,
                int shrink_axis_mask) {
            std::ostringstream oss;
            auto slice_size = std::min(std::min(begin.size(), end.size()), stride.size());
            oss << "[";
            for (size_t i = 0; i < slice_size; ++i) {
                if (i) oss << ", ";
                if (ellipsis_mask & (1 << i)) {
                    oss << "...";
                } else if (new_axis_mask & (1 << i)) {
                    oss << "None";
                } else if (shrink_axis_mask & (1 << i)) {
                    oss << begin[i];
                } else {
                    std::string begin_content = (begin_mask & (1 << i)) ? "" : std::to_string(begin[i]);
                    std::string end_content = (end_mask & (1 << i)) ? "" : std::to_string(end[i]);
                    std::string stride_content = stride[i] == 1 ? "" : std::to_string(stride[i]);
                    if (stride_content.empty()) {
                        oss << begin_content << ":" << end_content;
                    } else {
                        oss << begin_content << ":" << end_content << ":" << stride_content;
                    }
                }
            }
            oss << "]";
            return oss.str();
        }


        int SliceV3::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            load_params(stack);
            auto &x = stack[0];
            auto begin = m_begin;
            auto end = m_end;
            auto stride = m_stride;
            Shape y, in, out;
            auto succeed = infer_output(x.sizes(), y, begin, end, stride, m_begin_mask, m_end_mask, m_ellipsis_mask,
                                        m_new_axis_mask, m_shrink_axis_mask, in, out);

            if (!succeed) {
                TS_LOG_ERROR << "Can not stride slice on x=" << x.proto() << ", slice="
                             << slice_string(m_begin, m_end, m_stride,
                                             m_begin_mask, m_end_mask, m_ellipsis_mask, m_new_axis_mask,
                                             m_shrink_axis_mask) << eject;
            }

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), out);

            return 1;
        }

        int SliceV3::run(Stack &stack) {
            load_params(stack);
            auto &x = stack[0];
            auto begin = m_begin;
            auto end = m_end;
            auto stride = m_stride;
            Shape y, in, out;

            auto succeed = infer_output(x.sizes(), y, begin, end, stride, m_begin_mask, m_end_mask, m_ellipsis_mask,
                                        m_new_axis_mask, m_shrink_axis_mask, in, out);

            if (!succeed) {
                TS_LOG_ERROR << "Can not stride slice on x=" << x.proto() << ", slice="
                             << slice_string(m_begin, m_end, m_stride,
                                             m_begin_mask, m_end_mask, m_ellipsis_mask, m_new_axis_mask,
                                             m_shrink_axis_mask) << eject;
            }

            auto memory_device = running_memory_device();

            auto raw_in = stack[0].view(memory_device).reshape(in);
            auto &raw_out = *stack.push(x.dtype(), y, memory_device);

            slice(raw_in, begin.std(), end.std(), stride.std(), raw_out);

            raw_out = raw_out.reshape(out);

            return 1;
        }

        static std::vector<int32_t> tensor_long_to_int(const Tensor &tensor) {
            auto long_array = tensor::array::to_long(tensor);
            auto int_array = std::vector<int32_t>(long_array.size());

            for (size_t i = 0; i < int_array.size(); ++i) {
                auto v = long_array[i];
                if (v > 0 && v > INT_MAX) {
                    int_array[i] = INT_MAX;
                } else if (v < 0 && v < INT_MIN) {
                    int_array[i] = INT_MIN;
                } else {
                    int_array[i] = int32_t(v);
                }
            }

            return int_array;
        }

        void SliceV3::load_params(const Stack &stack) {
            // set m_begin, m_end, m_stride
            // set every mask
            TS_AUTO_CHECK(stack.size() >= 3);

            m_begin_mask = 0;
            m_end_mask = 0;
            m_ellipsis_mask = 0;
            m_new_axis_mask = 0;
            m_shrink_axis_mask = 0;

            auto &x = stack[0];
            auto dims = int32_t(x.dims());

            auto starts = tensor_long_to_int(stack[1]);
            auto ends = tensor_long_to_int(stack[2]);

            auto slice_size = int32_t(starts.size());
            TS_AUTO_CHECK(slice_size > 0);

            Shape axes;
            if (stack.size() > 3) {
                axes = tensor_long_to_int(stack[3]);
            } else {
                axes = Shape(starts.size());
                for (int32_t i = 0; i < slice_size; ++i) {
                    axes[i] = i;
                }
            }
            TS_AUTO_CHECK(axes.size() == starts.size());

            Shape steps;
            if (stack.size() > 4) {
                steps = tensor_long_to_int(stack[4]);
            } else {
                steps = Shape(slice_size, 1);
            }

            int32_t max_axis = INT32_MIN;
            int32_t pre_axis = INT32_MIN;
            for (int32_t i = 0; i < slice_size; ++i) {
                auto axis = axes[i];
                // axis in [-dims, dims)
                if (axis < -dims || axis >= dims) {
                    TS_LOG_ERROR << "axes must in [" << -dims << ", " << dims << "), but got " << to_string(axes)
                                 << eject;
                }
                if (axis < 0) axis += dims;
                if (axis <= pre_axis) {
                    TS_LOG_ERROR << "axes must in increasing rank, bug got " << to_string(axes) << eject;
                }
                pre_axis = axis;

                if (axis > max_axis) max_axis = axis;
            }
            auto max_dims = max_axis + 1;
            TS_AUTO_CHECK(max_dims <= dims);

            m_begin_mask = (1 << (max_dims + 1)) - 1;
            m_end_mask = (1 << (max_dims + 1)) - 1;

            m_begin = Shape(max_dims, 0);
            m_end = Shape(max_dims, 0);
            m_stride = Shape(max_dims, 1);

            for (int32_t i = 0; i < slice_size; ++i) {
                auto start = starts[i];
                auto end = ends[i];
                auto axis = axes[i];
                auto step = steps[i];

                m_begin[axis] = start;
                m_end[axis] = end;
                m_stride[axis] = step;

                m_begin_mask &= ~(1 << axis);
                m_end_mask &= ~(1 << axis);
            }
        }
    }
}
