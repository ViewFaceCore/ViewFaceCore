//
// Created by kier on 2019/1/25.
//

#ifndef TENSORSTACK_RUNTIME_IMAGE_FILTER_H
#define TENSORSTACK_RUNTIME_IMAGE_FILTER_H

#include <vector>
#include "core/tensor.h"
#include "utils/implement.h"
#include "program.h"

namespace ts {
    class Graph;
    class Workbench;
    class TS_DEBUG_API ImageFilter {
    public:
        using self = ImageFilter;

        using shared = std::shared_ptr<self>;

        enum class ResizeMethod : int32_t {
            BILINEAR = 0,
            BICUBIC = 1,
            NEAREST = 2,
        };

        ImageFilter();

        explicit ImageFilter(const ComputingDevice &device);

        ImageFilter(const self &) = delete;

        ImageFilter &operator=(const self &) = delete;

        void to_float();

        void scale(float f);

        void sub_mean(const std::vector<float> &mean);

        void div_std(const std::vector<float> &std);

        void resize(int width, int height, ResizeMethod method = ResizeMethod::BILINEAR);

        void resize(int short_side, ResizeMethod method = ResizeMethod::BILINEAR);

        void center_crop(int width, int height);

        void center_crop(int side);

        void channel_swap(const std::vector<int> &shuffle);

        void prewhiten();

        void to_chw();

        void letterbox(int width, int height, float outer_value = 0, ResizeMethod method = ResizeMethod::BILINEAR);

        void divided(int width, int height, float padding_value);
        
        void force_color();
        
        void force_gray();
        
        /**
         * ensure image to gray
         * @param scale [0.114, 0.587, 0.299] as BGR format may give
         */
        void force_gray(const std::vector<float> &scale);

        void norm_image(float epsilon);

        /**
         * Clear all set processor
         */
        void clear();

        /**
         * Compile all processor
         */
        void compile();

        /**
         * Do ImageFilter
         * @param image Supporting Int8 and Float,
         *              Shape is [height, width, channels]
         * @return Converted image
         */
        Tensor run(const Tensor &image);

        shared clone() const;

        const Graph &graph() const;

        Module::shared module() const;

        Program::shared program() const;

    private:
        class Implement;
        Declare<Implement> m_impl;

        explicit ImageFilter(const Implement &other);

        std::string serial_name() const;
    };
}


#endif //TENSORSTACK_RUNTIME_IMAGE_FILTER_H
