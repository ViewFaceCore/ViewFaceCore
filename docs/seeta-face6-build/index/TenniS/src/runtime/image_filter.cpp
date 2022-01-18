//
// Created by kier on 2019/1/25.
//

#include <runtime/image_filter.h>

#include "runtime/image_filter.h"

#include "utils/ctxmgr_lite.h"
#include "module/menu.h"

#include "backend/name.h"
#include "core/tensor_builder.h"
#include "runtime/workbench.h"

#include <numeric>

namespace ts {

    class ImageFilter::Implement {
    public:
        ComputingDevice m_computing_device;
        // Workbench::shared m_workbench;
        Program::shared m_program;
        Graph::shared m_graph;
        bool m_compiled = false;
    };

    ImageFilter::ImageFilter() : self(ComputingDevice(CPU)) {
    }

    ImageFilter::ImageFilter(const ComputingDevice &device) {
        m_impl->m_computing_device = device;
        this->clear();
    }

    void ImageFilter::clear() {
        m_impl->m_program.reset();
        m_impl->m_graph = std::make_shared<Graph>();
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        bubble::param(serial_name(), UINT8, {-1, -1, -1});    // add input param to graph
        m_impl->m_compiled = false;
    }

    void ImageFilter::compile() {
        if (m_impl->m_compiled) return;
        if (m_impl->m_graph->nodes().size() > 1) {
            Module::shared module = std::make_shared<Module>();
            module->load(*m_impl->m_graph);
            m_impl->m_program = Program::Compile(module, m_impl->m_computing_device);
        }
        m_impl->m_compiled = true;
    }

    class ShapeTransformer {
    public:
        Shape before(const Shape &shape) {
            m_base = shape;
            switch (shape.size()) {
                case 0:
                    TS_LOG_ERROR << "Can not transform empty shape." << eject;
                    break;
                case 1:
                    return {1, shape[0], 1, 1};
                case 2:
                    return {1, shape[0], shape[1], 1};
                case 3:
                    return {1, shape[0], shape[1], shape[2]};
                case 4:
                    return shape;
                default:
                    return {shape[0], shape[1], shape[2],
                            std::accumulate(shape.begin() + 3, shape.end(), 1, std::multiplies<int>())};
            }
            return Shape();
        };
        Shape after(const Shape &shape) {
            return shape;
        }

    private:
        Shape m_base;
    };

    Tensor ImageFilter::run(const Tensor &image) {
        if (!m_impl->m_compiled) this->compile();
        if (!m_impl->m_program) return image;

        Tensor nhwc_image = image;
        ShapeTransformer transformer;

        nhwc_image = nhwc_image.reshape(transformer.before(nhwc_image.sizes()));

        Workbench &bench = ctx::of<Workbench>::ref();

        auto outputs = bench.launch_offline(m_impl->m_program, {nhwc_image});

        auto output = outputs[0];

        output = output.reshape(transformer.after(output.sizes()));

        return output;
    }

    std::string ImageFilter::serial_name() const {
        return "_" + std::to_string(m_impl->m_graph->nodes().size());
    }

    void ImageFilter::to_float() {
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto top = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::to_float(), {top});
        (void)(node);
        m_impl->m_compiled = false;
    }

    void ImageFilter::scale(float f) {
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto lhs = m_impl->m_graph->nodes().back();
        auto rhs = bubble::data(serial_name(), tensor::build(FLOAT32, f));
        auto node = bubble::op(serial_name(), name::layer::mul(), {lhs, rhs});
        (void)(node);
        m_impl->m_compiled = false;
    }

    void ImageFilter::sub_mean(const std::vector<float> &mean) {
        auto mean_tensor = tensor::build(FLOAT32, {1, 1, 1, int(mean.size())}, mean);
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto lhs = m_impl->m_graph->nodes().back();
        auto rhs = bubble::data(serial_name(), mean_tensor);
        auto node = bubble::op(serial_name(), name::layer::sub(), {lhs, rhs});
        (void)(node);
        m_impl->m_compiled = false;
    }

    void ImageFilter::div_std(const std::vector<float> &std) {
        auto std_tensor = tensor::build(FLOAT32, {1, 1, 1, int(std.size())}, std);
        auto count = std_tensor.count();
        auto std_data = std_tensor.data<float>();
        for (int i = 0; i < count; ++i) {
            std_data[i] = 1.0f / std_data[i];
        }
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto lhs = m_impl->m_graph->nodes().back();
        auto rhs = bubble::data(serial_name(), std_tensor);
        auto node = bubble::op(serial_name(), name::layer::mul(), {lhs, rhs});
        (void)(node);
        m_impl->m_compiled = false;
    }

    void ImageFilter::resize(int width, int height, ResizeMethod method) {
        auto size_tensor = tensor::build(INT32, {-1, height, width, -1});
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto x = m_impl->m_graph->nodes().back();
        auto size = bubble::data(serial_name(), size_tensor);;
        auto node = bubble::op(serial_name(), name::layer::resize2d(), {x, size});
        node->set(name::type, tensor::from(int32_t(method)));  // set resize method
        (void)(node);
        m_impl->m_compiled = false;
    }

    void ImageFilter::resize(int shot_size, ResizeMethod method) {
        auto size_tensor = tensor::build(INT32, {shot_size});
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto x = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::nhwc_scale_resize2d(), {x});
        node->set(name::size, size_tensor);
        node->set(name::type, tensor::from(int32_t(method)));  // set resize method
        (void)(node);
        m_impl->m_compiled = false;
    }

    void ImageFilter::channel_swap(const std::vector<int> &shuffle) {
        auto shuffle_tensor = tensor::build(INT32, shuffle);
        auto dim_tensor = tensor::build(INT32, {3, });
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto x = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::dimshuffle(), {x});
        node->set(name::dim, dim_tensor);
        node->set(name::shuffle, shuffle_tensor);
        m_impl->m_compiled = false;
    }

    void ImageFilter::to_chw() {
        auto permute_tensor = tensor::build(INT32, {0, 3, 1, 2});
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto x = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::transpose(), {x});
        node->set(name::permute, permute_tensor);
        m_impl->m_compiled = false;
    }

    void ImageFilter::center_crop(int width, int height) {
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto x = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::nhwc_center_crop2d(), {x});
        node->set(name::size, tensor::build(INT32, {width, height}));  // set resize method
        (void)(node);
        m_impl->m_compiled = false;

    }

    ImageFilter::shared ImageFilter::clone() const {
        ImageFilter::shared dolly(new ImageFilter(*this->m_impl));
        return dolly;
    }

    ImageFilter::ImageFilter(const ImageFilter::Implement &other) {
        m_impl->m_computing_device = other.m_computing_device;
        this->clear();
        m_impl->m_program = other.m_program->clone();
        m_impl->m_compiled = true;
    }

    const Graph &ImageFilter::graph() const {
        return *m_impl->m_graph;
    }

    void ImageFilter::center_crop(int side) {
        center_crop(side, side);
    }

    void ImageFilter::prewhiten() {
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto x = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::prewhiten(), {x});
        m_impl->m_compiled = false;
    }

    Module::shared ImageFilter::module() const {
        Module::shared module = std::make_shared<Module>();
        module->load(*m_impl->m_graph);
        return module;
    }

    Program::shared ImageFilter::program() const {
        return m_impl->m_program;
    }

    void ImageFilter::letterbox(int width, int height, float outer_value, ResizeMethod method) {
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto x = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::nhwc_letterbox(), {x});
        node->set(name::size, tensor::build(INT32, {width, height}));
        node->set(name::type, tensor::build(INT32, int32_t(method)));
        node->set(name::outer_value, tensor::build(FLOAT32, outer_value));
        (void)(node);
        m_impl->m_compiled = false;
    }

    void ImageFilter::divided(int width, int height, float padding_value) {
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto x = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::divided(), {x});
        node->set(name::size, tensor::build(INT32, {1, height, width, 1}));
        node->set(name::padding_value, tensor::from<float>(padding_value));
        (void)(node);
        m_impl->m_compiled = false;
    }

    void ImageFilter::force_color() {
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto top = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::force_color(), {top});
        (void)(node);
        m_impl->m_compiled = false;
    }

    void ImageFilter::force_gray() {
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto top = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::force_gray(), {top});
        (void)(node);
        m_impl->m_compiled = false;
    }

    void ImageFilter::force_gray(const std::vector<float> &scale) {
        auto scale_tensor = tensor::build(FLOAT32, {int32_t(scale.size())}, scale);
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto top = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::force_gray(), {top});
        node->set(name::scale, scale_tensor);
        m_impl->m_compiled = false;
    }

    void ImageFilter::norm_image(float epsilon) {
        ctx::bind<Graph> _bind_graph(m_impl->m_graph.get());
        auto top = m_impl->m_graph->nodes().back();
        auto node = bubble::op(serial_name(), name::layer::norm_image(), {top});
        node->set(name::epsilon, tensor::from<float>(epsilon));
        m_impl->m_compiled = false;
    }
}
