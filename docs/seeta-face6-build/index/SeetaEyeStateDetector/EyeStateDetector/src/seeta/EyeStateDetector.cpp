#include "seeta/EyeStateDetector.h"
#include "seeta/common_alignment.h"

#include "seeta/ImageProcess.h"
#include "seeta/CommonStruct.h"
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

            struct {
                std::string version = "single";
                int height = 256;
                int width = 256;
                int channels = 3;
            } alignment;

            std::vector<orz::jug> pre_processor;

            struct {
                orz::jug tsm;
            } backbone;

            struct {
                bool normalize = true;
                int sqrt_times = 0;
            } post_processor;

            struct {
                float threshold = 0.05;
                struct {
                    std::string format = "HWC";
                    int height = 256;
                    int width = 256;
                    int channels = 3;
                } input;
                struct {
                    int size = 256;
                } output;
                orz::jug compare;   // an op like {"op": "dot"}, could be invalid
                orz::jug similarity;    // an op like {"op": "sigmoid", "params": [3.0, 7.0]} or {"op", "none"}, could be invalid
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

        class CompareEngine {
        public:
            using self = CompareEngine;
            using shared = std::shared_ptr<self>;

            virtual float compare(const float *lhs, const float *rhs, int size) = 0;

            static shared Load(const orz::jug &jug);
        };

        class CompareDot : public CompareEngine {
        public:
            using self = CompareDot;
            using supper = CompareEngine;
            using shared = std::shared_ptr<self>;

            float compare(const float *lhs, const float *rhs, int size) final {
                float sum = 0;
                for (int i = 0; i < size; ++i) {
                    sum += *lhs * *rhs;
                    ++lhs;
                    ++rhs;
                }
                return sum;
            }
        };

        CompareEngine::shared CompareEngine::Load(const orz::jug &jug) {
            if (jug.invalid(orz::Piece::DICT)) {
                ORZ_LOG(orz::ERROR) << "Model: /global/compare must be dict" << orz::crash;
            }
            auto op = orz::jug_get<std::string>(jug["op"], "");
            if (op.empty()) {
                ORZ_LOG(orz::ERROR) << R"(Model: /global/compare should be set like {"op": "dot"}.)" << orz::crash;
            }
            if (op == "dot") {
                return std::make_shared<CompareDot>();
            } else {
                ORZ_LOG(orz::ERROR) << "Model: /global/compare \"" << jug << "\" not supported." << orz::crash;
            }
            return nullptr;
        }

        class SimilarityEngine {
        public:
            using self = SimilarityEngine;
            using shared = std::shared_ptr<self>;

            virtual float similarity(float x) = 0;

            static shared Load(const orz::jug &jug);
        };

        class SimilarityNone : public SimilarityEngine {
        public:
            using self = SimilarityNone;
            using supper = SimilarityEngine;
            using shared = std::shared_ptr<self>;

            float similarity(float x) final { return std::max<float>(x, 0); }
        };

        class SimilaritySigmoid : public SimilarityEngine {
        public:
            using self = SimilaritySigmoid;
            using supper = SimilarityEngine;
            using shared = std::shared_ptr<self>;

            SimilaritySigmoid(float a = 0, float b = 1)
                : m_a(a), m_b(b) {}

            float similarity(float x) final {
                return 1 / (1 + std::exp(m_a - m_b * std::max<float>(x, 0)));
            }

        private:
            float m_a;
            float m_b;
        };

        SimilarityEngine::shared SimilarityEngine::Load(const orz::jug &jug) {
            if (jug.invalid(orz::Piece::DICT)) {
                ORZ_LOG(orz::ERROR) << "Model: /global/similarity must be dict" << orz::crash;
            }
            auto op = orz::jug_get<std::string>(jug["op"], "");
            if (op.empty()) {
                ORZ_LOG(orz::ERROR) << R"(Model: /global/similarity should be set like {"op": "none"}.)" << orz::crash;
            }
            if (op == "none") {
                return std::make_shared<SimilarityNone>();
            } else if (op == "sigmoid") {
                std::vector<float> params;
                try {
                    params = ModelParam::to_float_list(jug["params"]);
                } catch (...) {}
                if (params.size() != 2) {
                    ORZ_LOG(orz::ERROR) << R"(Model: /global/similarity "sigmoid" must set "params" like "{"op": "sigmoid", "params": [0, 1]}")" << orz::crash;
                }
                return std::make_shared<SimilaritySigmoid>(params[0], params[1]);
            } else {
                ORZ_LOG(orz::ERROR) << "Model: /global/similarity \"" << jug << "\" not supported." << orz::crash;
            }
            return nullptr;
        }

        ModelParam parse_model(const orz::jug &model) {
            ModelParam param;

            if (model.invalid(orz::Piece::DICT)) ORZ_LOG(orz::ERROR) << "Model: / must be dict" << orz::crash;

            auto pre_processor = model["pre_processor"];
            auto backbone = model["backbone"];
            auto post_processor = model["post_processor"];
            auto global = model["global"];
            auto alignment = model["alignment"];

            if (alignment.valid()) {
                if (alignment.valid(orz::Piece::DICT)) {
                    auto version = orz::jug_get<std::string>(alignment["version"], "single");
                    param.alignment.version = version;
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

            if (post_processor.valid()) {
                if (post_processor.valid(orz::Piece::DICT)) {
                    param.post_processor.normalize = orz::jug_get<bool>(post_processor["normalize"], true);
                    if (!param.post_processor.normalize) {
                        ORZ_LOG(orz::ERROR) << "Model: /post_processor/normalize must be true" << orz::crash;
                    }
                    param.post_processor.sqrt_times = orz::jug_get<int>(post_processor["sqrt_times"], param.post_processor.sqrt_times);
                } else {
                    ORZ_LOG(orz::ERROR) << "Model: /post_processor must be dict" << orz::crash;
                }
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
                        ORZ_LOG(orz::ERROR) << R"(Model: processor "div_std" must set "mean" like "{"op": "div_std", "std": [128, 128, 128]}")" << orz::crash;
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


        //////////////////////////////////////////////////////////
        class EyeStateDetector::Implement {
        public:
            Implement(const seeta::ModelSetting &setting);
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

            void set(EyeStateDetector::Property property, double value) {
                switch (property) {
                    default:
                        break;
                    /*
                    case EyeStateDetector::PROPERTY_THRESHOLD:
                    {
                        if (value < M_MIN_FACE_SIZE_POWER) value = M_MIN_FACE_SIZE_POWER;
                        m_param.global.threshold = float(value);
                        break;
                    }
                    case EyeStateDetector::PROPERTY_MIN_FACE_SIZE:
                        m_min_face_size = int32_t(value);
                        break;
                    case EyeStateDetector::PROPERTY_MAX_IMAGE_WIDTH:
                        m_max_image_width = int32_t(value);
                        break;
                    case EyeStateDetector::PROPERTY_MAX_IMAGE_HEIGHT:
                        m_max_image_height = int32_t(value);
                        break;
                    */
                    case EyeStateDetector::PROPERTY_NUMBER_THREADS:
                    {
                        if (value < 1) value = 1;
                        auto threads = int(value);
                        m_number_threads = threads;
                        m_bench.set_computing_thread_number(threads);
                        break;
                    }

                    case EyeStateDetector::PROPERTY_ARM_CPU_MODE:
                    {
                        set_cpu_affinity(int32_t(value));
                        break;
                    }

                }
            }

            double get(EyeStateDetector::Property property) const {
                switch (property) {
                    default:
                        return 0;
                    /*
                    case EyeStateDetector::PROPERTY_THRESHOLD:
                        return m_param.global.threshold;
                    case EyeStateDetector::PROPERTY_MIN_FACE_SIZE:
                        return m_min_face_size;
                    case EyeStateDetector::PROPERTY_MAX_IMAGE_WIDTH:
                        return m_max_image_width;
                    case EyeStateDetector::PROPERTY_MAX_IMAGE_HEIGHT:
                        return m_max_image_height;
                    */
                    case EyeStateDetector::PROPERTY_NUMBER_THREADS:
                        return m_number_threads;
                    case EyeStateDetector::PROPERTY_ARM_CPU_MODE:
                        return get_cpu_affinity();

                }
            }

        int detect(const seeta::Image &image); 
        std::vector<int> detect(const std::vector<seeta::Image> &images);
        public:
            ModelParam m_param;
            mutable Workbench m_bench;

            //static const int32_t M_MIN_FACE_SIZE_POWER = 20;
            //int32_t m_min_face_size = M_MIN_FACE_SIZE_POWER;
            //int32_t m_max_image_width = 2000;
            //int32_t m_max_image_height = 2000;
            int32_t m_number_threads = 4;
            int m_cpu_affinity = -1;
        };

        EyeStateDetector::Implement::Implement(const seeta::ModelSetting &setting) {
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


        // return max index, 0 for close, 1 for open, 2 for random, 3 other 
        int EyeStateDetector::Implement::detect(const seeta::Image &image) {
            //std::cout << "---height:" << image.height() << ",width:" << image.width() << ",channels:" << image.channels() << std::endl;
            auto tensor = tensor::build(UINT8, {1, image.height(), image.width(), image.channels()}, image.data());
            m_bench.input(0, tensor);
            m_bench.run();
            auto output = m_bench.output(0);
            output.sync_cpu();
            output = ts::api::tensor::cast(TS_FLOAT32, output);
            output = output.reshape({1, -1});

            int max = 0;

            if(output.size(1) < 4)
            {
                ORZ_LOG(orz::ERROR) << "EyeStateDetector output size(1) must >=4, cur:"  << output.size(1);
                return 3;
            }

            float * pdata = output.data<float>();
            for(int i=0; i<output.size(1); i++)  {
                if(pdata[i] > pdata[max]) {
                    max = i;
                } 
            }
          
            double openEyeThreshold = 0.75;
            double closeEyeThreshold = 0.1;
 
            if(max == 2)
            {
                max = 2;
            }else if(pdata[1] > openEyeThreshold)
            {
                max = 1;
            }else if(pdata[0] > closeEyeThreshold)
            {
                max = 0;
            }else
            {
                max = 3; 
            } 
            return max;
        }

        /*
        std::vector<int> EyeStateDetector::Implement::detect(const std::vector<seeta::Image> &images) {
            std::vector<int> maxs;
            if(images.size() <= 0) {
                return maxs; 
            }

            int number = images[0].height() * images[0].width() * images[0].channels();
            uint8_t * buf = new uint8_t[number * images.size()];
            uint8_t *ptr = buf;
            for(int i=0; i<images.size(); i++) {
                memcpy(ptr, images[i].data(), number);
                ptr += number; 
            }

            Shape shape({images.size(), images[0].height(), images[0].width(), images[0].channels()});
            auto tensor = tensor::build<uint8_t>(UINT8, shape, buf);
            delete [] buf;

            m_bench.input(0, tensor);
            m_bench.run();
            auto output = tensor::cast(FLOAT32, m_bench.output(0));

            maxs.resize(images.size());

            number = output.count() / images.size();

            int max = 0;
            float * pdata = output.data<float>();
            for(int i=0; i<maxs.size(); i++)  {
                max = 0;
                int nstep = i * number;
                for(int m = 1; m < number; m++) {
                    if(pdata[nstep  + m] > pdata[ nstep + max]) {
                        max = m;
                    }
                } 
                maxs[i] = max; 
            }
               
            return std::move(maxs);
        }
        */
    }

    //////////////////////////////////////////////////////
    EyeStateDetector::EyeStateDetector(const seeta::ModelSetting &setting)
        : m_impl(new Implement(setting)) {

    }

    EyeStateDetector::~EyeStateDetector() {
        delete m_impl;
    }


    void EyeStateDetector::Detect(const SeetaImageData& image, const SeetaPointF* points, EYE_STATE &leftstate, EYE_STATE &rightstate) {
       
        leftstate = rightstate = EYE_UNKNOWN; 
        if (!points) return;

        seeta::Image simage = image;

        //auto shape = seeta::face_meanshape(5, 0);
        //seeta::Landmarks cropMarks;

        //auto cropImage = seeta::crop_face(simage, shape, seeta::Landmarks(points, 5), seeta::BY_BICUBIC, shape.size, cropMarks);
        //double eyeSpan = sqrt(pow((cropMarks.points[1].x - cropMarks.points[0].x), 2) + pow((cropMarks.points[1].y - cropMarks.points[0].y), 2));
        double eyeSpan = sqrt(pow((points[1].x - points[0].x), 2) + pow((points[1].y - points[0].y), 2));
        //seeta::Size eyeSize(99, 99);

        seeta::Size eyeSize(102, 102);
        //·ÀÖ¹±ß½çÒç³ö
        int leftEyePointX = std::max(int(points[0].x - eyeSpan / 2), 0);
        int leftEyePointY = std::max(int(points[0].y - eyeSpan / 2), 0);
        leftEyePointX = std::min(int(image.width - eyeSpan / 2), leftEyePointX);
        leftEyePointY = std::min(int(image.height - eyeSpan / 2), leftEyePointY);

        double leftEyeSpanTemp = std::max(int(eyeSpan) , 1);
        int leftEyeSpanX = (leftEyePointX + leftEyeSpanTemp > image.width - 1) ? image.width - 1 - leftEyePointX: leftEyeSpanTemp;
        int leftEyeSpanY = (leftEyePointY + leftEyeSpanTemp > image.height - 1) ? image.height - 1 - leftEyePointY : leftEyeSpanTemp;
        leftEyeSpanTemp = std::min(int(leftEyeSpanX), int(leftEyeSpanY));
        

        int rightEyePointX = std::max(int(points[1].x - eyeSpan / 2), 0);
        int rightEyePointY = std::max(int(points[1].y - eyeSpan / 2), 0);
        rightEyePointX = std::min(int(image.width - eyeSpan / 2), rightEyePointX);
        rightEyePointY = std::min(int(image.height - eyeSpan / 2), rightEyePointY);


        double rightEyeSpanTemp = std::max(int(eyeSpan) , 1);
        int rightEyeSpanX = (rightEyePointX + rightEyeSpanTemp > image.width - 1) ? image.width - 1 - rightEyePointX: rightEyeSpanTemp;
        int rightEyeSpanY = (rightEyePointY + rightEyeSpanTemp > image.height - 1) ? image.height - 1 - rightEyePointY : rightEyeSpanTemp;
        rightEyeSpanTemp = std::min(int(rightEyeSpanX), int(rightEyeSpanY));

        //std::cout << "----x:" << leftEyePointX << ", y:" << leftEyePointY << ",width:" << leftEyeSpanTemp;
        //std::cout << "  x:" << rightEyePointX << ", y:" << rightEyePointY << ",width:" << rightEyeSpanTemp << std::endl << std::endl;

        seeta::Image leftEye = seeta::crop_resize(simage, seeta::Rect(leftEyePointX, leftEyePointY, leftEyeSpanTemp, leftEyeSpanTemp), eyeSize);

        seeta::Image rightEye = seeta::crop_resize(simage, seeta::Rect(rightEyePointX, rightEyePointY, rightEyeSpanTemp, rightEyeSpanTemp), eyeSize);

        leftstate = (EYE_STATE) m_impl->detect(leftEye);
        rightstate = (EYE_STATE) m_impl->detect(rightEye);
        
        return;

    }


    void EyeStateDetector::set(EyeStateDetector::Property property, double value) {
        m_impl->set(property, value);
    }

    double EyeStateDetector::get(EyeStateDetector::Property property) const {
        return m_impl->get(property);
    }


}
