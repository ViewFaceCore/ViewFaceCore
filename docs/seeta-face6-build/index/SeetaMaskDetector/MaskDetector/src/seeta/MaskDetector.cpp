#include "seeta/MaskDetector.h"

#include <orz/utils/log.h>
#include "seeta/CommonStruct.h"
#include "seeta/ImageProcess.h"

#include <api/cpp/tensorstack.h>
#include <api/cpp/module.h>

#include <orz/io/jug/jug.h>
#include <orz/io/i.h>
#include <orz/io/dir.h>
#include <orz/codec/json.h>
#include <fstream>

#ifdef SEETA_MODEL_ENCRYPT
#include "SeetaLANLock.h"
#include "hidden/SeetaLockFunction.h"
#include "hidden/SeetaLockVerifyLAN.h"
#include "orz/io/stream/filestream.h"
#endif


#include "model_helper.h"

#define VER_HEAD(x) #x "."
#define VER_TAIL(x) #x
#define GENERATE_VER(seq) FUN_MAJOR seq
#define FUN_MAJOR(x) VER_HEAD(x) FUN_MINOR
#define FUN_MINOR(x) VER_HEAD(x) FUN_SINOR
#define FUN_SINOR(x) VER_TAIL(x)

#define LIBRARY_VERSION GENERATE_VER( \
                                      (SEETA_POSE_ESTIMATOR_MAJOR_VERSION) \
                                      (SEETA_POSE_ESTIMATOR_MINOR_VERSION) \
                                      (SEETA_POSE_ESTIMATOR_SINOR_VERSION))

#define LIBRARY_NAME "PoseEstimator"

#define LOG_HEAD LIBRARY_NAME "(" LIBRARY_VERSION "): "

#include <orz/io/i.h>
#include <map>

#if SEETA_LOCK_SDK
#include <lock/macro.h>
#endif

namespace seeta {
    namespace v2 {
        using namespace ts::api;

        static int CLAMP(int a, int min, int max) {
            return std::max(min, std::min(max, a));
        }


        static SeetaRect V6toV5(const SeetaSize &limit, const SeetaRect &face) {
            /**
             * INFO: width scale: 1.1311
             * INFO: height scale: 1.13779
             * INFO: x shift: -0.0683691
             * INFO: y shift: -0.060302
             */
            float width_scale = 1.1311;
            float height_scale = 1.13779;
            float x_shift = -0.0683691;
            float y_shift = -0.060302;
            SeetaRect rect = face;

            rect.x += int(x_shift * rect.width);
            rect.y += int(y_shift * rect.height);
            rect.width = int(rect.width * width_scale);
            rect.height = int(rect.height * height_scale);
            int x1 = CLAMP(rect.x, 0, limit.width - 1);
            int y1 = CLAMP(rect.y, 0, limit.height - 1);
            int x2 = CLAMP(rect.x + rect.width - 1, 0, limit.width - 1);
            int y2 = CLAMP(rect.y + rect.height - 1, 0, limit.height - 1);
            int w = x2 - x1 + 1;
            int h = y2 - y1 + 1;

            rect.x = x1;
            rect.y = y1;
            rect.width = w;
            rect.height = h;

            return rect;
        }


        static std::string read_txt_file(std::ifstream &in) {
            std::ostringstream tmp;
            tmp << in.rdbuf();
            return tmp.str();
        }

        static orz::jug read_jug_from_json_or_sta(const std::string &filename) {
            std::ifstream ifs(filename, std::ios::binary);
            int32_t mark;
            ifs.read(reinterpret_cast<char *>( &mark ), 4);

            try {
                if (mark == orz::STA_MASK) {
                    return orz::jug_read(ifs);
                } else {
                    std::string json = read_txt_file(ifs);
                    return orz::json2jug(json);
                }
            }
            catch (const orz::Exception &) {
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
                float threshold = 0.5f;
                struct {
                    std::string format = "HWC";
                    int height = 128;
                    int width = 128;
                    int channels = 3;
                } input;
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

                param.global.threshold = orz::jug_get<float>(global["threshold"], param.global.threshold);
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
                    ORZ_LOG(orz::ERROR) << "Model: the " << i << "-th processor \"" << processor << "\" should be dict"
                                        << orz::crash;
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
                    }
                    catch (...) {}
                    if (mean.empty()) {
                        ORZ_LOG(orz::ERROR)
                                << R"(Model: processor "sub_mean" must set "mean" like "{"op": "sub_mean", "mean": [104, 117, 123]}")"
                                << orz::crash;
                    }
                    filter.sub_mean(mean);
                } else if (op == "center_crop") {
                    std::vector<int> size;
                    try {
                        size = ModelParam::to_int_list(processor["size"]);
                    }
                    catch (...) {}
                    if (size.empty()) {
                        ORZ_LOG(orz::ERROR)
                                << R"(Model: processor "center_crop" must set "mean" like "{"op": "center_crop", "size": [248, 248]}")"
                                << orz::crash;
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
                    }
                    catch (...) {}
                    if (size.empty()) {
                        ORZ_LOG(orz::ERROR)
                                << R"(Model: processor "resize" must set "mean" like "{"op": "resize", "size": [248, 248]}")"
                                << orz::crash;
                    }
                    if (size.size() == 1) {
                        filter.resize(size[0]);
                    } else {
                        filter.resize(size[0], size[1]);
                    }
                } else if (op == "prewhiten") {
                    filter.prewhiten();
                } else if (op == "channel_swap") {
                    std::vector<int> shuffle;
                    try {
                        shuffle = ModelParam::to_int_list(processor["shuffle"]);
                    }
                    catch (...) {}
                    if (shuffle.size() != 3) {
                        ORZ_LOG(orz::ERROR)
                                << R"(Model: processor "resize" must set "mean" like "{"op": "channel_swap", "shuffle": [2, 1, 0]}")"
                                << orz::crash;
                    }
                    filter.channel_swap(shuffle);
                } else {
                    ORZ_LOG(orz::ERROR) << "Model: processor \"" << processor << "\" not supported." << orz::crash;
                }
            }
        }

        static std::string to_string(const std::vector<int> &shape) {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i) oss << ", ";
                oss << shape[i];
            }
            oss << "]";
            return oss.str();
            (void) (to_string);
        }

        static float raw2degree(float raw) {
            return float((1.0 / (1.0 + std::exp(-raw))) * 180.0 - 90.0);
        }

        class MaskDetector::Implement {
        public:
            Implement(const seeta::ModelSetting &setting) {
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
                // ts_Workbench_setup_device(bench.get_raw());
                ImageFilter filter(device);

                build_filter(filter, param.pre_processor);
                bench.bind_filter(0, filter);

                this->m_param = param;
                this->m_bench = bench;

            }


            bool detect(const SeetaImageData &image, const SeetaRect &info, float &score) {
#ifdef SEETA_CHECK_AUTO_FUNCID
                SEETA_CHECK_AUTO_FUNCID("MaskDetector");
#endif
                score = 0;
                if (!image.data || image.channels != 3) {
                    return false;
                }

                SeetaSize facesize;
                facesize.width = image.width;
                facesize.height = image.height;
                SeetaRect facerect = V6toV5(facesize, info);

                seeta::Size size(m_param.global.input.height, m_param.global.input.width);
                seeta::Rect rect(facerect.x, facerect.y, facerect.width, facerect.height);


                seeta::Image cropped_face = seeta::crop_resize(image, rect, size);

                auto tensor = tensor::build(UINT8,
                                            {1, cropped_face.height(), cropped_face.width(), cropped_face.channels()},
                                            cropped_face.data());
                m_bench.input(0, tensor);
                m_bench.run();
                auto output = tensor::cast(FLOAT32, m_bench.output(0));

                score = output.data<float>(0);
                return score >= m_param.global.threshold;
            }

            ModelParam m_param;
            mutable Workbench m_bench;

        };


        //////////////////////////////////////

        MaskDetector::MaskDetector(const seeta::ModelSetting &setting)
                : m_impl(new Implement(setting)) {
        }

        MaskDetector::~MaskDetector() {
            delete m_impl;
        }


        bool MaskDetector::detect(const SeetaImageData &image, const SeetaRect &face, float *score) {
            float tmp;
            return m_impl->detect(image, face, score ? *score : tmp);
        }
    }
}

