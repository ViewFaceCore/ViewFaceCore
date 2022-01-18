//
// Created by kier on 19-4-24.
//

#include "seeta/FaceLandmarker.h"
#include "seeta/common_alignment.h"

#include <orz/utils/log.h>
#include <api/cpp/tensorstack.h>
#include <api/cpp/module.h>

#include <orz/io/jug/jug.h>
#include <orz/io/i.h>
#include <orz/io/dir.h>
#include <orz/codec/json.h>
#include <fstream>
#include <cfloat>
#include <cmath>

#ifdef SEETA_MODEL_ENCRYPT
#include "SeetaLANLock.h"
#include "hidden/SeetaLockFunction.h"
#include "hidden/SeetaLockVerifyLAN.h"
#endif

#include "orz/io/stream/filestream.h"
#include "model_helper.h"

namespace seeta {
    namespace v6 {
        using namespace ts::api;

        static std::string read_txt_file(std::ifstream &in) {
            std::ostringstream tmp;
            tmp << in.rdbuf();
            return tmp.str();
        }
        static orz::jug read_jug_from_json_or_sta(const std::string &filename) {
            std::ifstream ifs(filename, std::ios::binary);

            if (!ifs.is_open()) {
                ORZ_LOG(orz::ERROR) << "Can not access: " << filename << orz::crash;
            }

            int32_t mark;
            ifs.read(reinterpret_cast<char*>(&mark), 4);

            orz::jug model;

            try {
                if (mark == orz::STA_MASK) {
                    model = orz::jug_read(ifs);
                } else {
                    ifs.seekg(0, std::ios::beg);
                    std::string json = read_txt_file(ifs);
                    model = orz::json2jug(json, filename);

                }
            } catch (const orz::Exception &) {
                ORZ_LOG(orz::ERROR) << "Model must be sta or json file, given: " << filename << orz::crash;
                return orz::jug();
            }

            if (model.invalid()) {
                ORZ_LOG(orz::ERROR) << "File format error: " << filename << orz::crash;
            }

            return model;
        }

        static Module parse_tsm_module(const orz::jug &model, const std::string &root) {
            if (model.valid(orz::Piece::BINARY)) {
                auto binary = model.to_binary();
                BufferReader reader(binary.data(), binary.size());
                return Module::Load(reader);
            } else if (model.valid(orz::Piece::STRING)) {
                auto commands = orz::Split(model.to_string(), '@', 3);
                if (commands.size() != 3 || !commands[0].empty() || commands[1] != "file") {
                    ORZ_LOG(orz::ERROR) << R"(Model: /backbone/tsm must be "@file@..." or "@binary@...")" << orz::crash;
                }
                std::string path = root.empty() ? commands[2] : orz::Join({root, commands[2]}, orz::FileSeparator());
                return Module::Load(path);
            } else {
                ORZ_LOG(orz::ERROR) << R"(Model: /backbone/tsm must be "@file@..." or "@binary@...")" << orz::crash;
            }
            return Module();
        }

        struct ModelParam {
            ModelParam() = default;

            std::vector<orz::jug> pre_processor;

            std::vector<orz::jug> transform;

            struct {
                orz::jug tsm;
            } backbone;

            struct {
                std::vector<std::vector<int>> pickup;
            } post_processor;

            struct {
                int number;
                struct {
                    std::string format = "HWC";
                    int height = 112;
                    int width = 112;
                    int channels = 1;
                } input;
                bool occlusion = false;
            } global;

            static bool to_bool(const orz::jug &jug) {
                return jug.to_bool();
            }
            static std::vector<int> to_int_list(const orz::jug &jug) {
                if (jug.invalid(orz::Piece::LIST)) throw orz::Exception("jug must be list");
                std::vector<int> list(jug.size());
                for (size_t i = 0; i < list.size(); ++i) {
                    list[i] = jug[i].to_int();
                }
                return std::move(list);
            }
            static std::vector<std::vector<int>> to_int_list_list(const orz::jug &jug) {
                if (jug.invalid(orz::Piece::LIST)) throw orz::Exception("jug must be list");
                std::vector<std::vector<int>> list(jug.size());
                for (size_t i = 0; i < list.size(); ++i) {
                    list[i] = to_int_list(jug[i]);
                }
                return std::move(list);
            }
            static std::vector<float> to_float_list(const orz::jug &jug) {
                if (jug.invalid(orz::Piece::LIST)) throw orz::Exception("jug must be list");
                std::vector<float> list(jug.size());
                for (size_t i = 0; i < list.size(); ++i) {
                    list[i] = jug[i].to_float();
                }
                return std::move(list);
            }
            static std::vector<std::vector<float>> to_float_list_list(const orz::jug &jug) {
                if (jug.invalid(orz::Piece::LIST)) throw orz::Exception("jug must be list");
                std::vector<std::vector<float>> list(jug.size());
                for (size_t i = 0; i < list.size(); ++i) {
                    list[i] = to_float_list(jug[i]);
                }
                return std::move(list);
            }
        };

        ModelParam parse_model(const orz::jug &model) {
            ModelParam param;

            if (model.invalid(orz::Piece::DICT)) ORZ_LOG(orz::ERROR) << "Model: / must be dict" << orz::crash;

            auto transform = model["transform"];
            auto pre_processor = model["pre_processor"];
            auto backbone = model["backbone"];
            auto post_processor = model["post_processor"];
            auto global = model["global"];

            if (transform.valid()) {
                if (transform.valid(orz::Piece::LIST)) {
                    auto size = transform.size();
                    for (decltype(size) i = 0; i < size; ++i) {
                        param.transform.emplace_back(transform[i]);
                    }
                } else {
                    ORZ_LOG(orz::ERROR) << "Model: /transform must be list" << orz::crash;
                }
            }

            if (pre_processor.valid()) {
                if (pre_processor.valid(orz::Piece::LIST)) {
                    auto size = pre_processor.size();
                    for (decltype(size) i = 0; i < size; ++i) {
                        param.pre_processor.emplace_back(pre_processor[i]);
                    }
                } else {
                    ORZ_LOG(orz::ERROR) << "Model: /pre_processor must be list" << orz::crash;
                }
            }

            if (backbone.valid(orz::Piece::DICT)) {
                auto tsm = backbone["tsm"];
                if (tsm.invalid()) {
                    ORZ_LOG(orz::ERROR) << R"(Model: /backbone/tsm must be "@file@..." or "@binary@...")" << orz::crash;
                }
                param.backbone.tsm = tsm;
            } else {
                ORZ_LOG(orz::ERROR) << "Model: /backbone must be dict" << orz::crash;
            }

            if (post_processor.valid()) {
                if (post_processor.valid(orz::Piece::DICT)) {
                    auto pickup = post_processor["pickup"];
                    if (pickup.valid()) {
                        if (pickup.valid(orz::Piece::LIST)) {
                            try {
                                auto &store = param.post_processor.pickup;
                                auto size = pickup.size();
                                for (int i = 0; i < size; ++i) {
                                    store.push_back(ModelParam::to_int_list(pickup[i]));
                                }
                            } catch (...) {}
                        } else {
                            ORZ_LOG(orz::ERROR) << "Model: /post_processor/pickup must be list of list" << orz::crash;
                        }
                    }
                } else {
                    ORZ_LOG(orz::ERROR) << "Model: /post_processor must be dict" << orz::crash;
                }
            }

            if (global.valid(orz::Piece::DICT)) {
                auto input = global["input"];
                if (input.invalid(orz::Piece::DICT)) ORZ_LOG(orz::ERROR) << "Model: /global/input must be dict" << orz::crash;

                decltype(param.global.input) param_input;
                param_input.format = orz::jug_get<std::string>(input["format"], param_input.format);
                param_input.height = orz::jug_get<int>(input["height"], param_input.height);
                param_input.width = orz::jug_get<int>(input["width"], param_input.width);
                param_input.channels = orz::jug_get<int>(input["channels"], param_input.channels);
                param.global.input = param_input;

                if (param_input.format != "HWC") {
                    ORZ_LOG(orz::ERROR) << "Model: /global/input/format must be HWC" << orz::crash;
                }

                param.global.number = orz::jug_get(global["number"], 0);
                if (param.global.number <= 0) {
                    ORZ_LOG(orz::ERROR) << "Model: /global/number must greater than 0" << orz::crash;
                }

                param.global.occlusion = orz::jug_get<bool>(global["occlusion"], false);
            } else {
                ORZ_LOG(orz::ERROR) << "Model: /global must be dict" << orz::crash;
            }

            return param;
        }

        Device to_ts_device(const seeta::ModelSetting &setting) {
            switch (setting.get_device()) {
                case seeta::ModelSetting::Device::AUTO:
                    return Device("cpu");
                case seeta::ModelSetting::Device::CPU:
                    return Device("cpu");
                case seeta::ModelSetting::Device::GPU:
                    return Device("gpu", setting.id);
                default:
                    return Device("cpu");
            }
        }

        static void build_filter(ImageFilter &filter, const std::vector<orz::jug> &pre_processor) {
            filter.clear();
            for (size_t i = 0; i < pre_processor.size(); ++i) {
                auto &processor = pre_processor[i];
                if (processor.invalid(orz::Piece::DICT)) {
                    ORZ_LOG(orz::ERROR) << "Model: the " << i << "-th processor \"" << processor << "\" should be dict" << orz::crash;
                }
                auto op = orz::jug_get<std::string>(processor["op"], "");
                if (op.empty()) {
                    ORZ_LOG(orz::ERROR) << R"(Model: processor should be set like {"op": "to_float"}.)" << orz::crash;
                }
                if (op == "to_float") {
                    filter.to_float();
                } else if (op == "to_chw") {
                    filter.to_chw();
                } else if (op == "force_gray") {
                    filter.force_gray();
                } else if (op == "scale") {
                    float scale = FLT_MAX;
                    scale = orz::jug_get<float>(processor["scale"], scale);
                    if (scale == FLT_MAX) {
                        ORZ_LOG(orz::ERROR) << R"(Model: processor "scale" must set "scale" like "{"op": "scale", "scale": 0.0039}")" << orz::crash;
                    }
                    filter.scale(scale);
                } else if (op == "sub_mean") {
                    std::vector<float> mean;
                    try {
                        mean = ModelParam::to_float_list(processor["mean"]);
                    } catch (...) {}
                    if (mean.empty()) {
                        ORZ_LOG(orz::ERROR) << R"(Model: processor "sub_mean" must set "mean" like "{"op": "sub_mean", "mean": [104, 117, 123]}")" << orz::crash;
                    }
                    filter.sub_mean(mean);
                }  else if (op == "div_std") {
                    std::vector<float> std_value;
                    try {
                        std_value = ModelParam::to_float_list(processor["std"]);
                    } catch (...) {}
                    if (std_value.empty()) {
                        ORZ_LOG(orz::ERROR) << R"(Model: processor "div_std" must set "std" like "{"op": "div_std", "std": [128, 128, 128]}")" << orz::crash;
                    }
                    filter.div_std(std_value);
                } else if (op == "center_crop") {
                    std::vector<int> size;
                    try {
                        size = ModelParam::to_int_list(processor["size"]);
                    } catch (...) {}
                    if (size.empty()) {
                        ORZ_LOG(orz::ERROR) << R"(Model: processor "center_crop" must set "mean" like "{"op": "center_crop", "size": [248, 248]}")" << orz::crash;
                    }
                    if (size.size() == 1) {
                        filter.center_crop(size[0]);
                    } else {
                        filter.center_crop(size[0], size[1]);
                    }
                } else if (op == "resize") {
                    std::vector<int> size;
                    try {
                        size = ModelParam::to_int_list(processor["size"]);
                    } catch (...) {}
                    if (size.empty()) {
                        ORZ_LOG(orz::ERROR) << R"(Model: processor "resize" must set "mean" like "{"op": "resize", "size": [248, 248]}")" << orz::crash;
                    }
                    if (size.size() == 1) {
                        filter.resize(size[0]);
                    } else {
                        filter.resize(size[0], size[1]);
                    }
                } else if (op == "prewhiten") {
                    filter.prewhiten();
                }  else if (op == "channel_swap") {
                    std::vector<int> shuffle;
                    try {
                        shuffle = ModelParam::to_int_list(processor["shuffle"]);
                    } catch (...) {}
                    if (shuffle.size() != 3) {
                        ORZ_LOG(orz::ERROR) << R"(Model: processor "resize" must set "mean" like "{"op": "channel_swap", "shuffle": [2, 1, 0]}")" << orz::crash;
                    }
                    filter.channel_swap(shuffle);
                }  else if (op == "norm_image") {
                    auto epsilon = orz::jug_get<float>(processor["epsilon"], 1e-5f);
                    filter.norm_image(epsilon);
                } else {
                    ORZ_LOG(orz::ERROR) << "Model: processor \"" << processor << "\" not supported." << orz::crash;
                }
            }
        }

        class SeetaRectF {
        public:
            SeetaRectF() = default;
            SeetaRectF(float x, float y, float w, float h)
                    : x(x), y(y), w(w), h(h) {}
            SeetaRectF(const SeetaRect &x)
                    : x(x.x), y(x.y), w(x.width), h(x.height) {}

            float x = 0;
            float y = 0;
            float w = 0;
            float h = 0;
        };

        class RectTransformer {
        public:
            virtual SeetaRectF transform(const SeetaRectF &x) = 0;

            virtual ~RectTransformer() = default;
        };

        class RectShiftTransformer : public RectTransformer {
        public:
            explicit RectShiftTransformer(float a, float b) : a(a), b(b) {}

            SeetaRectF transform(const SeetaRectF &x) final {
                return {x.x + x.w * a, x.y + x.h * b, x.w, x.h};
            }

        private:
            float a;
            float b;
        };

        class RectScaleTransformer : public RectTransformer {
        public:
            explicit RectScaleTransformer(float a, float b) : a(a), b(b) {}

            SeetaRectF transform(const SeetaRectF &x) final {
                return {x.x, x.y, x.w * a, x.h * b};
            }

        private:
            float a;
            float b;
        };

        class RectExpandTransformer : public RectTransformer {
        public:
            explicit RectExpandTransformer(float a, float b) : a(a), b(b) {}

            SeetaRectF transform(const SeetaRectF &x) final {
                // "x-=w*a;y-=h*b;w+=w*a*2;h+=h*b*2"
                return {x.x - x.w * a, x.y - x.h * b, x.w + x.w * a * 2, x.h + x.h * b * 2};
            }

        private:
            float a;
            float b;
        };

        class RectSquareTransformer : public RectTransformer {
        public:
            explicit RectSquareTransformer() {}

            SeetaRectF transform(const SeetaRectF &x) final {
                // d=max(w,h);x-=(d-w)/2;y-=(d-h);w=h=d;
                auto d = std::max(x.w, x.h);
                return {x.x - (d - x.w) / 2, x.y - (d - x.h) / 2, d, d};
            }
        };

        class TransformEngine {
        public:
            using self = TransformEngine;
            using shared = std::shared_ptr<self>;

            static shared Load(const std::vector<orz::jug> &transform) {
                auto shared_engine = std::make_shared<TransformEngine>();
                TransformEngine &engine = *shared_engine;
                auto &core = engine.m_core;
                for (size_t i = 0; i < transform.size(); ++i) {
                    auto &processor = transform[i];
                    if (processor.invalid(orz::Piece::DICT)) {
                        ORZ_LOG(orz::ERROR) << "Model: the " << i << "-th transform processor \"" << processor << "\" should be dict" << orz::crash;
                    }
                    auto op = orz::jug_get<std::string>(processor["op"], "");
                    if (op.empty()) {
                        ORZ_LOG(orz::ERROR) << R"(Model: transform processor should be set like {"op": "square"}.)" << orz::crash;
                    }
                    if (op == "shift") {
                        std::vector<float> param;
                        try {
                            param = ModelParam::to_float_list(processor["param"]);
                        } catch (...) {}
                        if (param.size() != 2) {
                            ORZ_LOG(orz::ERROR) << R"(Model: transform processor "shift" must set "param" length 2 like "{"op": "shift", "param": [0, 0]}")" << orz::crash;
                        }
                        core.emplace_back(std::make_shared<RectShiftTransformer>(param[0], param[1]));
                    } else if (op == "scale") {
                        std::vector<float> param;
                        try {
                            param = ModelParam::to_float_list(processor["param"]);
                        } catch (...) {}
                        if (param.size() != 2) {
                            ORZ_LOG(orz::ERROR) << R"(Model: transform processor "scale" must set "param" length 2 like "{"op": "scale", "param": [0, 0]}")" << orz::crash;
                        }
                        core.emplace_back(std::make_shared<RectScaleTransformer>(param[0], param[1]));
                    } else if (op == "expand") {
                        std::vector<float> param;
                        try {
                            param = ModelParam::to_float_list(processor["param"]);
                        } catch (...) {}
                        if (param.size() != 2) {
                            ORZ_LOG(orz::ERROR) << R"(Model: transform processor "expand" must set "param" length 2 like "{"op": "expand", "param": [0, 0]}")" << orz::crash;
                        }
                        core.emplace_back(std::make_shared<RectExpandTransformer>(param[0], param[1]));
                    } else if (op == "square") {
                        core.emplace_back(std::make_shared<RectSquareTransformer>());
                    } else {
                        ORZ_LOG(orz::ERROR) << "Model: transform processor \"" << processor << "\" not supported." << orz::crash;
                    }
                }
                return shared_engine;
            }

            SeetaRectF transform(const SeetaRectF &x) {
                auto rect = x;
                for (auto &core : m_core) {
                    rect = core->transform(rect);
                }
                return rect;
            }

        private:
            std::vector<std::shared_ptr<RectTransformer>> m_core;
        };

        class LandmarksWithMask {
        public:
            std::vector<SeetaPointF> points;
            std::vector<int32_t> masks;
        };

        class FaceLandmarker::Implement {
        public:
            explicit Implement(const seeta::ModelSetting &setting);

            Implement(const Implement &other);

            LandmarksWithMask mark(const SeetaImageData &image, const SeetaRect &face) const;

            Tensor crop_face(const SeetaImageData &image, const SeetaRectF &face) const;

        public:
            ModelParam m_param;
            mutable Workbench m_bench;

            TransformEngine::shared m_transform;

            mutable Tensor m_padding;
        };

        FaceLandmarker::Implement::Implement(const seeta::ModelSetting &setting) {
            auto &model = setting.get_model();
            if (model.size() != 1) {
                ORZ_LOG(orz::ERROR) << "Must have 1 model." << orz::crash;
            }
			
			auto jug = get_model_jug(model[0].c_str());
            auto param = parse_model(jug);

            // parse tsm module
            std::string root = orz::cut_path_tail(model[0]);
            auto tsm = parse_tsm_module(param.backbone.tsm, root);
            // add image filter
            auto device = to_ts_device(setting);
            auto bench = Workbench::Load(tsm, device);
            ts_Workbench_setup_device(bench.get_raw());
            ImageFilter filter(device);

            build_filter(filter, param.pre_processor);
            bench.bind_filter(0, filter);

            this->m_transform = TransformEngine::Load(param.transform);

            this->m_param = param;
            this->m_bench = bench;

            this->m_padding = Tensor(INT32, {3, 2});
        }

        Tensor FaceLandmarker::Implement::crop_face(const SeetaImageData &image, const SeetaRectF &face) const {
            auto source = tensor::build(UINT8, {image.height, image.width, image.channels}, image.data);
            auto padding_data = m_padding.data<int32_t >();
            /*
            H = image.shape[0]
            W = image.shape[1]
            x0, y0, x1, y1 = rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]
            padding = [[-y0, y1 - H], [-x0, x1 - W], [0, 0]]
             */
            auto H = image.height;
            auto W = image.width;
            auto x0 = int(std::round(face.x));
            auto y0 = int(std::round(face.y));
            auto x1 = int(std::round(face.x + face.w));
            auto y1 = int(std::round(face.y + face.h));

            padding_data[0] = -y0;
            padding_data[1] = y1 - H;
            padding_data[2] = -x0;
            padding_data[3] = x1 - W;
            padding_data[4] = 0;
            padding_data[5] = 0;

            return intime::pad(source, m_padding);
        }

        SeetaPointF average_point(const float *data, int N, const std::vector<int> &select) {
            SeetaPointF point = {0, 0};
            int count = 0;
            for (auto i : select) {
                if (i >= N) continue;
                point.x += data[2 * i];
                point.y += data[2 * i + 1];
                ++count;
            }
            point.x = point.x / count;
            point.y = point.y / count;
            return point;
        }

        LandmarksWithMask
        FaceLandmarker::Implement::mark(const SeetaImageData &image, const SeetaRect &face) const {
            // transform face
            auto fixed_face = m_transform->transform(face);
            m_bench.setup_context();

            auto patch = crop_face(image, fixed_face);
            // detector points
            m_bench.input(0, patch);
            m_bench.run();
            auto points_tensor = m_bench.output(0);
            points_tensor = tensor::cast(FLOAT32, points_tensor).reshape({m_param.global.number, -1});

            auto N = points_tensor.size(0);
            auto S = points_tensor.size(1);
            auto points_data = points_tensor.data<float>();

            LandmarksWithMask landmarks;

            // may pickup
            std::vector<SeetaPointF> points;
            if (m_param.post_processor.pickup.empty()) {
                for (int i = 0; i < N; ++i) {
                    points.push_back(seeta::PointF(points_data[i * S], points_data[i * S + 1]));
                }
            } else {
                for (auto &select : m_param.post_processor.pickup) {
                    points.push_back(average_point(points_data, N, select));
                }
            }

            std::vector<int32_t> masks;
            if (S > 2 && m_param.post_processor.pickup.empty()) {
                if (S != 4) {
                    ORZ_LOG(orz::ERROR) << "Unrecognized output shape: [" << N << ", " << S << "]" << orz::crash;
                }
                masks.resize(N);
                for (int i = 0; i < N; ++i) {
                    bool mask = points_data[i * S + 2] < points_data[i * S + 3];
                    masks[i] = mask ? 1 : 0;
                }
            } else {
                masks.resize(points.size(), 0);
            }

            for (auto &point : points) {
                point.x = fixed_face.x + fixed_face.w * point.x;
                point.y = fixed_face.y + fixed_face.h * point.y;
            }

            landmarks.points = points;
            landmarks.masks = masks;

            return landmarks;
        }

        FaceLandmarker::Implement::Implement(const FaceLandmarker::Implement &other) {
            *this = other;
            this->m_bench = this->m_bench.clone();
            this->m_padding = this->m_padding.clone();
        }
    }

    FaceLandmarker::FaceLandmarker(const SeetaModelSetting &setting)
        : m_impl(new Implement(setting)) {

    }

    FaceLandmarker::~FaceLandmarker() {
        delete m_impl;
    }

    int FaceLandmarker::number() const {
        if (m_impl->m_param.post_processor.pickup.empty()) {
            return m_impl->m_param.global.number;
        }
        return int(m_impl->m_param.post_processor.pickup.size());
    }

    void FaceLandmarker::mark(const SeetaImageData &image, const SeetaRect &face, SeetaPointF *points) const {
        auto result = m_impl->mark(image, face);
        for (size_t i = 0; i < result.points.size(); ++i) {
            points[i] = result.points[i];
        }
    }

    FaceLandmarker::FaceLandmarker(const FaceLandmarker::self *other)
            : m_impl(nullptr) {
        if (other == nullptr) {
            ORZ_LOG(orz::ERROR) << "Parameter 1 can not be nullptr." << orz::crash;
        }
        m_impl = new Implement(*other->m_impl);
    }

    void
    FaceLandmarker::mark(const SeetaImageData &image, const SeetaRect &face, SeetaPointF *points, int32_t *mask) const {
        auto result = m_impl->mark(image, face);
        for (size_t i = 0; i < result.points.size(); ++i) {
            points[i] = result.points[i];
            mask[i] = result.masks[i];
        }
    }
}