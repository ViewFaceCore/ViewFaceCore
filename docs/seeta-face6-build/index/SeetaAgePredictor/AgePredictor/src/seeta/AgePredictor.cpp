#include "seeta/AgePredictor.h"

#include "seeta/common_alignment.h"
#include "seeta/ImageProcess.h"

#include <orz/utils/log.h>
#include <api/cpp/tensorstack.h>
#include <api/cpp/module.h>

#include <orz/io/jug/jug.h>
#include <orz/io/i.h>
#include <orz/io/dir.h>
#include <orz/codec/json.h>
#include <fstream>
#include <cfloat>

#ifdef SEETA_MODEL_ENCRYPT
#include "SeetaLANLock.h"
#include "hidden/SeetaLockFunction.h"
#include "hidden/SeetaLockVerifyLAN.h"
#endif

#include "orz/io/stream/filestream.h"
#include "model_helper.h"

namespace seeta{
    namespace v6{

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

            struct {
                int height = 256;
                int width = 256;
                int channels = 3;
            } alignment;

            std::vector<orz::jug> pre_processor;

            struct {
                orz::jug tsm;
            } backbone;

            struct {
                struct {
                    std::string format = "HWC";
                    int height = 256;
                    int width = 256;
                    int channels = 3;
                } input;
                struct {
                    int size = 88;
                } output;
            } global;
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
            auto alignment = model["alignment"];

            if (alignment.valid()) {
                if (alignment.valid(orz::Piece::DICT)) {
                    param.alignment.width = orz::jug_get<int>(alignment["width"], param.alignment.width);
                    param.alignment.height = orz::jug_get<int>(alignment["height"], param.alignment.height);
                    param.alignment.channels = orz::jug_get<int>(alignment["channels"], param.alignment.channels);
                } else {
                    ORZ_LOG(orz::ERROR) << "Model: /alignment must be dict" << orz::crash;
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

            if (global.valid(orz::Piece::DICT)) {
                auto input = global["input"];
                if (input.invalid(orz::Piece::DICT)) ORZ_LOG(orz::ERROR) << "Model: /global/input must be dict" << orz::crash;
                auto output = global["output"];
                if (output.invalid(orz::Piece::DICT)) ORZ_LOG(orz::ERROR) << "Model: /global/output must be dict" << orz::crash;

                decltype(param.global.input) param_input;
                param_input.format = orz::jug_get<std::string>(input["format"], param_input.format);
                param_input.height = orz::jug_get<int>(input["height"], param_input.height);
                param_input.width = orz::jug_get<int>(input["width"], param_input.width);
                param_input.channels = orz::jug_get<int>(input["channels"], param_input.channels);
                param.global.input = param_input;

                auto param_output_size = output["size"];
                param.global.output.size = orz::jug_get(output["size"], 0);
                if (param.global.output.size <= 0) {
                    ORZ_LOG(orz::ERROR) << "Model: /global/output/size must greater than 0" << orz::crash;
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
            (void)(to_string);
        }

        class AgePredictor::Implement{
            public:

             Implement(const seeta::ModelSetting &setting);

             Implement(const Implement &other);

             bool PredictAge(const SeetaImageData &image, int &age) const;

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

            void set(AgePredictor::Property property, double value) {
                switch (property) {
                    default:
                        break;
                    case AgePredictor::PROPERTY_NUMBER_THREADS:
                    {
                        if (value < 1) value = 1;
                        auto threads = int(value);
                        m_number_threads = threads;
                        m_bench.set_computing_thread_number(threads);
                        break;
                    }

                    case AgePredictor::PROPERTY_ARM_CPU_MODE:
                    {
                        set_cpu_affinity(int32_t(value));
                        break;
                    }

                }
            }

            double get(AgePredictor::Property property) const {
                switch (property) {
                    default:
                        return 0;
                    case AgePredictor::PROPERTY_NUMBER_THREADS:
                        return m_number_threads;
                    case AgePredictor::PROPERTY_ARM_CPU_MODE:
                        return get_cpu_affinity();

                }
            }

        public:
            ModelParam m_param;
            mutable Workbench m_bench;

            int32_t m_number_threads = 4;
            int m_cpu_affinity = -1;
        };

        AgePredictor::Implement::Implement(const seeta::ModelSetting &setting) {
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

        AgePredictor::Implement::Implement(const AgePredictor::Implement &other) {
            *this = other;
            this->m_bench = this->m_bench.clone();
        }

       bool AgePredictor::Implement::PredictAge(const SeetaImageData &image, int &age) const
       {
        if (image.height != m_param.global.input.height ||
                image.width != m_param.global.input.width ||
                image.channels != m_param.global.input.channels)
                return false;

        auto tensor = tensor::build(UINT8, {1, image.height, image.width, image.channels}, image.data);
        m_bench.input(0, tensor);
        m_bench.run();
        auto output = tensor::cast(FLOAT32,m_bench.output(0));
        auto output_size = m_param.global.output.size;
        if (output.count() != output_size) {
            ORZ_LOG(orz::ERROR) << "Extracted features size must be "
                                << output_size << " vs. " << output.count() << " given.";
            return false;
        }

        #ifdef _DEBUG
        int _count = output_size;
        std::cout << "LOG: Predict count: " << _count << std::endl;
        std::cout << "LOG: Predict result: ";
        for (int i = 0; i < _count; ++i)
        {
            if (i) std::cout << ", ";
            std::cout << output.data<float>(i);
        }
        std::cout << std::endl;
        #endif	// _DEBUG

        int start = 0;
        float age_f = 0;
        for (int i = 0; i < output_size; ++i)
        {
            age_f += output.data<float>(i) * (start + i);
        }

        age = static_cast<int>(age_f + 0.5);
        return true;
       }

       AgePredictor::AgePredictor(const SeetaModelSetting &setting)
        : m_impl(new Implement(setting)) {

    }

    AgePredictor::~AgePredictor() {
        delete m_impl;
    }

    int AgePredictor::GetCropFaceWidth() const{
        return m_impl->m_param.alignment.width;
    }

    int AgePredictor::GetCropFaceHeight() const{
        return m_impl->m_param.alignment.height;
    }

    int AgePredictor::GetCropFaceChannels() const{
        return m_impl->m_param.alignment.channels;
    }

    bool AgePredictor::CropFace(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face) const {
        if (face.width != GetCropFaceWidth() || face.height != GetCropFaceHeight() || face.channels != GetCropFaceChannels()) return false;
        auto meanshape = seeta::face_meanshape(5, 1);
        meanshape = seeta::resize(meanshape, { 191, 191 });
        auto cropped_face = seeta::crop_face(image, meanshape, seeta::Landmarks(points, 5), seeta::BY_LINEAR, { 256, 256 });

        cropped_face = seeta::resize(cropped_face, seeta::Size(GetCropFaceWidth(), GetCropFaceHeight()));

        cropped_face.copy_to(face.data);
        return true;
    }

    bool AgePredictor::PredictAge(const SeetaImageData &image, int &age) const
    {
       return m_impl->PredictAge(image, age);
    }

    bool AgePredictor::PredictAgeWithCrop(const SeetaImageData &image, const SeetaPointF *points, int &age) const
    {
        SeetaImageData cropped_face;
        cropped_face.width = GetCropFaceWidth();
        cropped_face.height = GetCropFaceHeight();
        cropped_face.channels = GetCropFaceChannels();
        std::unique_ptr<unsigned char[]> data(new unsigned char[cropped_face.width * 
                        cropped_face.height * cropped_face.channels]);
        cropped_face.data = data.get();

        if (!CropFace(image, points, cropped_face))
        {
            return false;
        }

        return PredictAge(cropped_face, age);
    }

    void AgePredictor::set(AgePredictor::Property property, double value) {
        m_impl->set(property, value);
    }

    double AgePredictor::get(AgePredictor::Property property) const {
        return m_impl->get(property);
    }

    }
}