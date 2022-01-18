//
// Created by kier on 2019/1/12.
//

#ifndef TENSORSTACK_BACKEND_COMMON_STRUCTURE_H
#define TENSORSTACK_BACKEND_COMMON_STRUCTURE_H

#include <cstdint>

namespace ts {
    template <typename T>
    class Aspect2D {
    public:
        using Dtype = T;

        Aspect2D() = default;
        Aspect2D(Dtype top, Dtype bottom, Dtype left, Dtype right)
                : top(top), bottom(bottom), left(left), right(right) {}

        Dtype top = 0;
        Dtype bottom = 0;
        Dtype left = 0;
        Dtype right = 0;
    };

    template <typename T>
    inline bool operator==(const Aspect2D<T> &lhs, const Aspect2D<T> &rhs) {
        return lhs.top == rhs.top && lhs.bottom == rhs.bottom && lhs.left == rhs.left && lhs.right == rhs.right;
    }

    template <typename T>
    inline bool operator!=(const Aspect2D<T> &lhs, const Aspect2D<T> &rhs) {
        return !operator==(lhs, rhs);
    }

    template <typename T>
    class Form2D {
    public:
        using Dtype = T;

        Form2D() = default;
        Form2D(Dtype height, Dtype width)
                : height(height), width(width) {}

        Dtype height = 0;
        Dtype width = 0;
    };

    template <typename T>
    inline bool operator==(const Form2D<T> &lhs, const Form2D<T> &rhs) {
        return lhs.height == rhs.height && lhs.width == rhs.width;
    }

    template <typename T>
    inline bool operator!=(const Form2D<T> &lhs, const Form2D<T> &rhs) {
        return !operator==(lhs, rhs);
    }

    using Padding2D = Aspect2D<int32_t>;

    using Stride2D = Form2D<int32_t>;

    using KSize2D = Form2D<int32_t>;

    using Dilation2D = Form2D<int32_t>;

    using Size2D = Form2D<int32_t>;

    enum DataFormat{
        DATA_NCHW = 0,
        DATA_NHWC = 1,
    };

    enum Conv2DFormat {
        FORMAT_NCHW = 0,
        FORMAT_NHWC = 1,
    };

    enum WinogradConv2DMode {
        F6X6_3X3 = 0,
        F2X2_3X3 = 1,
    };

    enum class Pooling2DType : int {
        MAX = 0,
        AVG = 1,
    };

    enum class Padding2DType : int {
        BLACK = 0,
        COPY = 1,
        LOOP = 2,
        WHITE = 3,
    };

    enum class Resize2DType : int {
        LINEAR = 0,
        CUBIC = 1,
        NEAREST = 2,
        HARD = 3,
    };

    enum class Affine_Sample2DType : int {
        LINEAR = 0,
        CUBIC = 1,
        NEAREST = 2,
        HARD = 3,
    };
}

#endif //TENSORSTACK_BACKEND_COMMON_STRUCTURE_H
