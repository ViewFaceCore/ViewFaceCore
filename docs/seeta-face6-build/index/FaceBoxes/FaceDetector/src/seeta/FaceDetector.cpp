
#include "seeta/FaceDetector.h"

#include <orz/utils/log.h>
#include <api/cpp/tensorstack.h>
#include <api/cpp/module.h>

#include <orz/io/jug/jug.h>
#include <orz/io/i.h>
#include <orz/io/dir.h>
#include <orz/codec/json.h>
#include <fstream>
#include <array>
#include <cmath>

#ifdef SEETA_MODEL_ENCRYPT
#include "SeetaLANLock.h"
#include "hidden/SeetaLockFunction.h"
#include "hidden/SeetaLockVerifyLAN.h"
#include "orz/io/stream/filestream.h"
#endif

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
            ifs.seekg(0, std::ios::beg);

            try {
                if (mark == orz::STA_MASK) {
                    return orz::sta_read(ifs);
                } else {
                    std::string json = read_txt_file(ifs);
                    return orz::json2jug(json, filename);
                }
            } catch (const orz::Exception &) {
                ORZ_LOG(orz::ERROR) << "Model must be sta or json file, given: " << filename << orz::crash;
                return orz::jug();
            }
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
            struct {
                orz::jug tsm;
            } backbone;
            struct {
                std::vector<float> variance;
                bool clip;
                std::vector<int> steps;
                std::vector<std::vector<float>> aspect_ratios;
                std::vector<std::vector<int>> min_sizes;
            } prior_box;
            struct {
                float threshold = 0.3;
                int top_k = 5000;
                int keep_top_k = 750;
            } nms;
            struct {
                float threshold = 0.05;
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
            auto pre_processor = model["pre_processor"];
            auto backbone = model["backbone"];
            auto prior_box = model["prior_box"];
            auto nms = model["nms"];
            auto global = model["global"];
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

            if (prior_box.valid(orz::Piece::DICT)) {
                try {
                    param.prior_box.variance = ModelParam::to_float_list(prior_box["variance"]);
                } catch (const orz::Exception &) {
                    ORZ_LOG(orz::ERROR) << "Model: /prior_box/variance must be list of float" << orz::crash;
                }
                try {
                    param.prior_box.clip = ModelParam::to_bool(prior_box["clip"]);
                } catch (const orz::Exception &) {
                    ORZ_LOG(orz::ERROR) << "Model: /prior_box/clip must be boolean" << orz::crash;
                }
                try {
                    param.prior_box.steps = ModelParam::to_int_list(prior_box["steps"]);
                } catch (const orz::Exception &) {
                    ORZ_LOG(orz::ERROR) << "Model: /prior_box/steps must be list of int" << orz::crash;
                }
                try {
                    param.prior_box.aspect_ratios = ModelParam::to_float_list_list(prior_box["aspect_ratios"]);
                } catch (const orz::Exception &) {
                    ORZ_LOG(orz::ERROR) << "Model: /prior_box/aspect_ratios must be list of float list" << orz::crash;
                }
                try {
                    param.prior_box.min_sizes = ModelParam::to_int_list_list(prior_box["min_sizes"]);
                } catch (const orz::Exception &) {
                    ORZ_LOG(orz::ERROR) << "Model: /prior_box/min_sizes must be list of int list" << orz::crash;
                }
            } else {
                ORZ_LOG(orz::ERROR) << "Model: /prior_box must be dict" << orz::crash;
            }

            if (nms.valid(orz::Piece::DICT)) {
                auto threshold = nms["threshold"];
                param.nms.threshold = orz::jug_get<float>(threshold, -1);
                if (param.nms.threshold < 0) {
                    ORZ_LOG(orz::ERROR) << "Model: /nms/threshold must be float in [0, 1]" << orz::crash;
                }
                param.nms.top_k = orz::jug_get<int>(nms["top_k"], param.nms.top_k);
                param.nms.keep_top_k = orz::jug_get<int>(nms["keep_top_k"], param.nms.keep_top_k);
            } else {
                ORZ_LOG(orz::ERROR) << "Model: /nms must be dict" << orz::crash;
            }

            if (global.valid(orz::Piece::DICT)) {
                auto threshold = global["threshold"];
                param.global.threshold = orz::jug_get<float>(threshold, -1);
                if (param.global.threshold < 0) {
                    ORZ_LOG(orz::ERROR) << "Model: /global/threshold must be float in [0, 1]" << orz::crash;
                }
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
                } else if (op == "sub_mean") {
                    std::vector<float> mean;
                    try {
                        mean = ModelParam::to_float_list(processor["mean"]);
                    } catch (...) {}
                    if (mean.empty()) {
                        ORZ_LOG(orz::ERROR) << R"(Model: processor "sub_mean" must set "mean" like "{"op": "sub_mean", "mean": [104, 117, 123]}")" << orz::crash;
                    }
                    filter.sub_mean(mean);
                } else {
                    ORZ_LOG(orz::ERROR) << "Model: processor \"" << processor << "\" not supported." << orz::crash;
                }
            }
        }

        class PriorBox {
        public:
            PriorBox() = default;
            PriorBox(float cx, float cy, float s_kx, float s_ky)
                : cx(cx), cy(cy), s_kx(s_kx), s_ky(s_ky) {}

            float cx;
            float cy;
            float s_kx;
            float s_ky;
        };

        class Product {
        public:
            using self = Product;

            Product(int M, int N) : M(M), N(N) {}

            class Iterator {
            public:
                using self = Iterator;

                Iterator(int M, int N, int m, int n)
                    : M(M), N(N), m(m), n(n) {}

                std::array<int, 2> operator*() const {
                    return {m, n};
                }

                Iterator &operator++() {
                    increase();
                    return *this;
                }

                const Iterator operator++(int) {
                    self copy = *this;
                    increase();
                    return copy;
                }

                bool operator==(const Iterator &it) const {
                    return m == it.m && n == it.n;
                }

                bool operator!=(const Iterator &it) const {
                    return !operator==(it);
                }

                void increase() {
                    ++n;
                    if (n >= N) {
                        n = 0;
                        ++m;
                    }
                }

            private:
                int M;
                int N;
                int m;
                int n;
            };

            Iterator begin() const {
                return Iterator(M, N, 0, 0);
            }

            Iterator end() const {
                return Iterator(M, N, M, 0);
            }

        private:
            int M;
            int N;
        };

        struct BindingBox {
            BindingBox() = default;
            BindingBox(float x1, float y1, float x2, float y2, float score = 0)
                : x1(x1), y1(y1), x2(x2), y2(y2), score(score) {}

            float x1;
            float y1;
            float x2;
            float y2;
            float score;
        };

        static inline BindingBox decode_single(const float *location, const float *priors, const float *variance, float score = 0) {
            /**
                boxes = torch.cat((
                    priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                    priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
                boxes[:, :2] -= boxes[:, 2:] / 2
                boxes[:, 2:] += boxes[:, :2]
                return boxes
             */
            BindingBox box;
            auto box_data = &box.x1;
            box_data[0] = priors[0] + location[0] * variance[0] * priors[2];
            box_data[1] = priors[1] + location[1] * variance[0] * priors[3];
            box_data[2] = priors[2] * std::exp(location[2] * variance[1]);
            box_data[3] = priors[3] * std::exp(location[3] * variance[1]);

            box_data[0] -= box_data[2] / 2;
            box_data[1] -= box_data[3] / 2;

            box_data[2] += box_data[0];
            box_data[3] += box_data[1];

            box_data[4] = score;

            return box;
        }

        class FaceDetector::Implement {
        public:
            Implement(const seeta::ModelSetting &setting);

            Implement(const Implement &impl) {
                *this = impl;
                this->m_bench = this->m_bench.clone();
                this->m_size = this->m_size.clone();
            }

            SeetaFaceInfoArray detect(const SeetaImageData &image) const;

            /**
             *
             * @param box_dimension IntMatrix
             * @param image_size
             * @return
             */
            std::vector<PriorBox> prior_box_forward(
                    const Tensor &box_dimension,
                    const std::array<int, 2> &image_size,
                    int tips = 100) const;

            std::vector<BindingBox> decode(
                    const Tensor &location,
                    const std::vector<PriorBox> &priors,
                    const std::vector<float> &variance,
                    const Tensor &confidence,
                    float threshold,
                    int top_k) const;

            void set(FaceDetector::Property property, double value) {
                switch (property) {
                    default:
                        break;
                    case FaceDetector::PROPERTY_THRESHOLD:
                        m_param.global.threshold = float(value);
                        break;
                    case FaceDetector::PROPERTY_MIN_FACE_SIZE:
                    {
                        if (value < M_MIN_FACE_SIZE_POWER) value = M_MIN_FACE_SIZE_POWER;
                        m_min_face_size = int32_t(value);
                        break;
                    }
                    case FaceDetector::PROPERTY_MAX_IMAGE_WIDTH:
                        m_max_image_width = int32_t(value);
                        break;
                    case FaceDetector::PROPERTY_MAX_IMAGE_HEIGHT:
                        m_max_image_height = int32_t(value);
                        break;
                    case FaceDetector::PROPERTY_NUMBER_THREADS:
                    {
                        if (value < 1) value = 1;
                        auto threads = int(value);
                        m_number_threads = threads;
                        m_bench.set_computing_thread_number(threads);
                        break;
                    }
                    case FaceDetector::PROPERTY_ARM_CPU_MODE:
                        set_cpu_affinity(int32_t(value));
                        break;
                }
            }

            double get(FaceDetector::Property property) const {
                switch (property) {
                    default:
                        return 0;
                    case FaceDetector::PROPERTY_THRESHOLD:
                        return m_param.global.threshold;
                    case FaceDetector::PROPERTY_MIN_FACE_SIZE:
                        return m_min_face_size;
                    case FaceDetector::PROPERTY_MAX_IMAGE_WIDTH:
                        return m_max_image_width;
                    case FaceDetector::PROPERTY_MAX_IMAGE_HEIGHT:
                        return m_max_image_height;
                    case FaceDetector::PROPERTY_NUMBER_THREADS:
                        return m_number_threads;
                    case FaceDetector::PROPERTY_ARM_CPU_MODE:
                        return get_cpu_affinity();
                }
            }

            int get_cpu_affinity() const {
                return m_cpu_affinity;
            }

            void set_cpu_affinity(int level) {
                switch (level) {
                    case 0:
                        m_bench.set_cpu_mode(CpuPowerMode::BIG_CORE);
                        break;
                    case 1:
                        m_bench.set_cpu_mode(CpuPowerMode::LITTLE_CORE);
                        break;
                    case 2:
                        m_bench.set_cpu_mode(CpuPowerMode::BALANCE);
                        break;
                    default:
                        level = -1;
                        break;
                }
                m_cpu_affinity = level;
            }

            float min_scale(const SeetaImageData &image) const {
                auto scale1 = float(M_MIN_FACE_SIZE_POWER) / m_min_face_size;
                auto scale2 = float(m_max_image_width) / image.width;
                auto scale3 = float(m_max_image_height) / image.height;
                auto scale = std::min(std::min(scale1, scale2), scale3);
                return std::min<float>(1.0f, scale);
            }

        private:
            ModelParam m_param;
            mutable Workbench m_bench;

            mutable std::vector<SeetaFaceInfo> m_pre_faces;

            static const int32_t M_MIN_FACE_SIZE_POWER = 20;

            int32_t m_min_face_size = M_MIN_FACE_SIZE_POWER;
            int32_t m_max_image_width = 2000;
            int32_t m_max_image_height = 2000;

            int32_t m_number_threads = 4;

            mutable Tensor m_size = nullptr;  // Int[4]

            int m_cpu_affinity = -1;
        };

        FaceDetector::Implement::Implement(const seeta::ModelSetting &setting) {
            auto &model = setting.get_model();
            if (model.size() != 1) {
                ORZ_LOG(orz::ERROR) << "Must have 1 model." << orz::crash;
            }
			
			auto jug = get_model_jug(model[0].c_str());


            auto param = parse_model(jug);

            // checking param
            if (param.prior_box.variance.size() != 2) {
                ORZ_LOG(orz::ERROR) << "Model: /prior_box/variance size must be 2" << orz::crash;
            }

            // parse tsm module
            std::string root = orz::cut_path_tail(model[0]);
            auto tsm = parse_tsm_module(param.backbone.tsm, root);
            // add image filter
            auto device = to_ts_device(setting);
            auto bench = Workbench::Load(tsm, device);
            ImageFilter filter(device);

            build_filter(filter, param.pre_processor);
            bench.bind_filter(0, filter);

            this->m_param = param;
            this->m_bench = bench;

            const static int32_t size_init[4] = {-1, -1, -1, -1};
            this->m_size = Tensor(INT32, {4}, size_init);

            this->m_bench.set_computing_thread_number(m_number_threads);
        }

        float IoU(const BindingBox &w1, const BindingBox &w2)
        {
            auto xOverlap = std::max<float>(0, std::min(w1.x2 - 1, w2.x2 - 1) - std::max(w1.x1, w2.x1) + 1);
            auto yOverlap = std::max<float>(0, std::min(w1.y2 - 1, w2.y2 - 1) - std::max(w1.y1, w2.y1) + 1);
            auto intersection = xOverlap * yOverlap;

            auto w1_width = w1.x2 - w1.x1;
            auto w1_height = w1.y2 - w1.y1;
            auto w2_width = w2.x2 - w2.x1;
            auto w2_height = w2.y2 - w2.y1;

            auto unio = w1_width * w1_height + w2_width * w2_height - intersection;
            return float(intersection) / unio;
        }

        std::vector<BindingBox> NMS_sorted(std::vector<BindingBox> &winList, float threshold)
        {
            if (winList.size() == 0)
                return winList;
            std::vector<bool> flag(winList.size(), false);
            for (size_t i = 0; i < winList.size(); i++)
            {
                if (flag[i]) continue;
                for (size_t j = i + 1; j < winList.size(); j++)
                {
                    if (IoU(winList[i], winList[j]) > threshold) flag[j] = true;
                }
            }
            std::vector<BindingBox> ret;
            for (size_t i = 0; i < winList.size(); i++)
            {
                if (!flag[i]) ret.push_back(winList[i]);
            }
            return ret;
        }


        SeetaFaceInfoArray FaceDetector::Implement::detect(const SeetaImageData &image) const {
            float im_scale = min_scale(image);

            m_bench.setup_context();

            // get faces
            auto tensor = tensor::build(UINT8, {1, image.height, image.width, image.channels}, image.data);

            if (im_scale < 0.99) {
                m_size.data<int32_t>(1) = int32_t(image.height * im_scale);
                m_size.data<int32_t>(2) = int32_t(image.width * im_scale);
                tensor = intime::resize2d(tensor, m_size);
            }

            auto input_height = tensor.size(1);
            auto input_width = tensor.size(2);

            m_bench.input(0, tensor);
            m_bench.run();

            auto location = m_bench.output(0);
            auto confidence = m_bench.output(1);
            auto detection_dimension = m_bench.output(2);

            location = tensor::cast(FLOAT32, location).reshape({-1, 4});
            confidence = tensor::cast(FLOAT32, confidence);
            detection_dimension = tensor::cast(INT32, detection_dimension);

            auto priors = prior_box_forward(detection_dimension, {input_height, input_width}, location.size(0));
            auto boxes = decode(location, priors, m_param.prior_box.variance, confidence, m_param.global.threshold, m_param.nms.top_k);

            // do nms, SeetaNet version
            boxes = NMS_sorted(boxes, m_param.nms.threshold);

            // score boxes
            for (auto &box : boxes) {
                box.x1 = box.x1 * image.width;
                box.y1 = box.y1 * image.height;
                box.x2 = box.x2 * image.width;
                box.y2 = box.y2 * image.height;
            }

            m_pre_faces.resize(boxes.size());
            for (size_t i = 0; i < boxes.size(); ++i) {
                auto &face = m_pre_faces[i];
                auto &box = boxes[i];
                face.pos.x = int(box.x1);
                face.pos.y = int(box.y1);
                face.pos.width = int(box.x2 - box.x1);
                face.pos.height = int(box.y2 - box.y1);

                if (face.pos.x < 0) face.pos.x = 0;
                if (face.pos.y < 0) face.pos.y = 0;
                if (face.pos.width >= image.width) face.pos.width = image.width - 1;
                if (face.pos.height >= image.height) face.pos.height = image.height - 1;

                face.score = box.score;
            }

            // map faces
            SeetaFaceInfoArray faces;
            faces.data = m_pre_faces.data();
            faces.size = decltype(faces.size)(m_pre_faces.size());
            return faces;
        }

        std::vector<PriorBox> FaceDetector::Implement::prior_box_forward(const Tensor &box_dimension,
                                                                         const std::array<int, 2> &image_size, int tips) const {
            std::vector<PriorBox> mean;
            mean.reserve(tips);
            auto steps = m_param.prior_box.steps;
            auto box_n = box_dimension.size(0);
            for (int k = 0; k < box_n; ++k) {
                auto f = box_dimension.data<int32_t>() + 2 * k;
                auto &min_sizes = m_param.prior_box.min_sizes[k];
                Product product_i_j(f[0], f[1]);
                for (auto i_j : product_i_j) {
                    int i = i_j[0];
                    int j = i_j[1];
                    for (auto &min_size : min_sizes) {
                        auto s_kx = float(min_size) / image_size[1];
                        auto s_ky = float(min_size) / image_size[0];
                        if (min_size == 32) {
                            float fi = i, fj = j;
                            std::vector<float> dense_cx = {fj + 0.0f, fj + 0.25f, fj + 0.5f, fj + 0.75f};
                            std::vector<float> dense_cy = {fi + 0.0f, fi + 0.25f, fi + 0.5f, fi + 0.75f};
                            std::for_each(dense_cx.begin(), dense_cx.end(),
                                          [&](float &x) { x = x * steps[k] / image_size[1]; });
                            std::for_each(dense_cy.begin(), dense_cy.end(),
                                          [&](float &y) { y = y * steps[k] / image_size[0]; });
                            Product product_y_x(int(dense_cy.size()), int(dense_cx.size()));
                            for (auto y_x : product_y_x) {
                                auto cy = dense_cy[y_x[0]];
                                auto cx = dense_cx[y_x[1]];
                                mean.emplace_back(cx, cy, s_kx, s_ky);
                            }
                        } else if (min_size == 64) {
                            float fi = i, fj = j;
                            std::vector<float> dense_cx = {fj + 0.0f, fj + 0.5f};
                            std::vector<float> dense_cy = {fi + 0.0f, fi + 0.5f};
                            std::for_each(dense_cx.begin(), dense_cx.end(),
                                          [&](float &x) { x = x * steps[k] / image_size[1]; });
                            std::for_each(dense_cy.begin(), dense_cy.end(),
                                          [&](float &y) { y = y * steps[k] / image_size[0]; });
                            Product product_y_x(int(dense_cy.size()), int(dense_cx.size()));
                            for (auto y_x : product_y_x) {
                                auto cy = dense_cy[y_x[0]];
                                auto cx = dense_cx[y_x[1]];
                                mean.emplace_back(cx, cy, s_kx, s_ky);
                            }
                        } else {
                            float fi = i, fj = j;
                            auto cx = (fj + 0.5) * steps[k] / image_size[1];
                            auto cy = (fi + 0.5) * steps[k] / image_size[0];
                            mean.emplace_back(cx, cy, s_kx, s_ky);
                        }
                    }
                }
            }

            if (m_param.prior_box.clip) {
                auto count = mean.size() * 4;
                auto begin = &mean[0].cx;
                auto end = begin + count;
                std::for_each(begin, end, [&](float &f) {
                    f = f > 1 ? 1 : (f < 0 ? 0 : f);
                });
            }

            return mean;
        }

        std::vector<BindingBox>
        FaceDetector::Implement::decode(const Tensor &location, const std::vector<PriorBox> &priors,
                                        const std::vector<float> &variance, const Tensor &confidence, float threshold,
                                        int top_k) const {
            std::vector<BindingBox> boxes;
            auto N = location.size(0);
            if (size_t(N) != priors.size()) {
                ORZ_LOG(orz::ERROR) << "Internal error with location and priors mismatch ("
                                    << N << " vs. " << priors.size() << ")";
                return boxes;
            }
            auto heap_comparer = [](const BindingBox &a, const BindingBox &b) {
                return a.score > b.score;
            };
            auto variance_data = variance.data();
            auto cast_top_k = size_t(top_k);
            for (int i = 0; i < N; ++i) {
                auto location_data = location.data<float>() + i * 4;
                auto priors_data = &priors[i].cx;
                auto score = confidence.data<float>(i * 2 + 1);
                if (score < threshold) continue;
                if (boxes.size() < cast_top_k) {
                    boxes.insert(boxes.end(), decode_single(location_data, priors_data, variance_data, score));
                    std::push_heap(boxes.begin(), boxes.end(), heap_comparer);
                } else {
                    auto &box = boxes.front();
                    if (box.score > score) continue;
                    std::pop_heap(boxes.begin(), boxes.end(), heap_comparer);
                    boxes.back() = decode_single(location_data, priors_data, variance_data, score);
                    std::push_heap(boxes.begin(), boxes.end(), heap_comparer);
                }
            }
            std::sort_heap(boxes.begin(), boxes.end(), heap_comparer);
            return std::move(boxes);
        }
    }
    FaceDetector::FaceDetector(const SeetaModelSetting &setting)
        : m_impl(new Implement(setting)) {
        // m_impl->set(PROPERTY_THRESHOLD, 0.5);
    }

    FaceDetector::~FaceDetector() {
        delete m_impl;
    }

    SeetaFaceInfoArray FaceDetector::detect(const SeetaImageData &image) const {
        return m_impl->detect(image);
    }

    void FaceDetector::set(FaceDetector::Property property, double value) {
        m_impl->set(property, value);
    }

    double FaceDetector::get(FaceDetector::Property property) const {
        return m_impl->get(property);
    }

    FaceDetector::FaceDetector(const FaceDetector::self *other)
        : m_impl(nullptr) {
        if (other == nullptr) {
            ORZ_LOG(orz::ERROR) << "Parameter 1 can not be nullptr." << orz::crash;
        }
        m_impl = new Implement(*other->m_impl);
    }
}
