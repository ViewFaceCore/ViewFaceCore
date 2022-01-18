//
// Created by kier on 2019/3/4.
//

#ifndef TENSORSTACK_TEST_CASE_HPP
#define TENSORSTACK_TEST_CASE_HPP

#include <string>
#include <map>
#include <fstream>
#include <algorithm>
#include <cmath>

#include <core/tensor.h>
#include <module/io/fstream.h>
#include <global/operator_factory.h>
#include <global/fp16_operator_factory.h>
#include <module/bubble.h>
#include <core/tensor_builder.h>
#include <runtime/stack.h>
#include <core/device_context.h>
#include <utils/ctxmgr_lite.h>
#include <runtime/workbench.h>
#include <backend/name.h>

#include "utils/box.h"
#include "utils/platform.h"

#if TS_PLATFORM_OS_WINDOWS

#include <direct.h>
#include <io.h>

#define ACCESS ::_access
#define MKDIR(a) ::_mkdir((a))
#define GETCWD(buffer, length) ::_getcwd((buffer), (length))
#define CHDIR(path) ::_chdir(path)

#include <Windows.h>
#include <sys/stat.h>
#ifdef min
#undef min
#endif

#elif TS_PLATFORM_OS_LINUX || TS_PLATFORM_OS_MAC || TS_PLATFORM_OS_IOS

#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <fstream>

#define ACCESS ::access
#define MKDIR(a) ::mkdir((a),0755)
#define GETCWD(buffer, length) ::getcwd((buffer), (length))
#define CHDIR(path) ::chdir(path)

#endif

namespace ts {
    static inline bool mkdir_core(const std::string &dir) {
        int miss = ACCESS(dir.c_str(), 0);
        if (miss) {
            int failed = MKDIR(dir.c_str());
            if (failed) {
                return false;
            }
        }
        return true;
    }

    static inline bool mkdir(const std::string &dir) {
        auto path = Split(dir, "\\/");
        for (size_t i = 1; i <= path.size(); ++i) {
            if (path[i - 1].empty()) continue;
            auto local_path = Join(std::vector<std::string>(path.begin(), path.begin() + i), "/");
            if (!mkdir_core(local_path)) return false;
        }
        return true;
    }

    static inline std::string plot_line(const std::vector<Tensor> &input, const std::vector<Tensor> &output) {
        std::ostringstream oss;
        oss << "{";
        for (size_t i = 0; i < input.size(); ++i) {
            auto &tensor = input[i];
            if (i) oss << ", ";
            if (tensor.fields_count() == 1)
                oss << tensor.proto();
            else
                oss << TensorPrototype(tensor);
        }
        oss << "} -> {";
        for (size_t i = 0; i < output.size(); ++i) {
            auto &tensor = output[i];
            if (i) oss << ", ";
            if (tensor.fields_count() == 1)
                oss << tensor.proto();
            else
                oss << TensorPrototype(tensor);
        }
        oss << "}";
        return oss.str();
    }

    static inline std::string plot_line(const std::vector<Tensor> &input, const TensorPrototype &output) {
        std::ostringstream oss;
        oss << "{";
        for (size_t i = 0; i < input.size(); ++i) {
            auto &tensor = input[i];
            if (i) oss << ", ";
            if (tensor.fields_count() == 1)
                oss << tensor.proto();
            else
                oss << TensorPrototype(tensor);
        }
        oss << "} -> ";
        oss << output;
        return oss.str();
    }

    static inline void diff(const Tensor &x, const Tensor &y, float &max, float &avg) {
        auto float_x = tensor::cast(FLOAT32, x);
        auto float_y = tensor::cast(FLOAT32, y);
        auto count = std::min(x.count(), y.count());
        auto float_x_data = float_x.data<float>();
        auto float_y_data = float_y.data<float>();
        float local_max = 0;
        float local_sum = 0;
        for (int i = 0; i < count; ++i) {
            auto diff = std::fabs(float_x_data[i] - float_y_data[i]);
            local_sum += diff;
            if (local_max < diff) local_max = diff;
        }
        max = local_max;
        avg = local_sum / count;
    }

    static inline std::ostream &plot_tensor(std::ostream &log, const Tensor &x) {
        int width = x.dims() == 0 ? 1 : x.sizes().back();
        for (int i = 0; i < width; ++i) {
            log << "--";
        }
        log << std::endl;
        auto float_x = tensor::cast(FLOAT32, x);
        int count = x.count();
        auto x_data= float_x.data<float>();
        int line_count = 0;
        for (int i = 0; i < count; ++i) {
            if (line_count >= width) {
                log << std::endl;
                line_count = 0;
            }
            log << x_data[i] << " ";
            ++line_count;

        }
        return log;
    }

    class TestCase {
    public:
        std::string op;
        std::string name;
        int output_count;
        int param_count;
        int input_count;
        std::map<std::string, Tensor> param;
        std::map<int, Tensor> input;
        std::map<int, Tensor> output;

        TestCase() = default;

        TestCase(const TestCase &other)
            : op(other.op), name(other.name)
            , output_count(other.output_count)
            , param_count(other.param_count)
            , input_count(other.input_count)
            , param(other.param), input(other.input), output(other.output) {
        }
        TestCase(TestCase &&other) {
            *this = std::move(other);
        }

        TestCase &operator=(const TestCase &other) {
            op = other.op;
            name = other.name;
            output_count = other.output_count;
            param_count = other.param_count;
            input_count = other.input_count;
            param = other.param;
            input = other.input;
            output = other.output;
            m_log.str(other.m_log.str());
            return *this;
        }

        /*
        TestCase &operator=(TestCase &&other) {
            op = std::move(other.op);
            name = std::move(other.name);
            output_count = other.output_count;
            param_count = other.param_count;
            input_count = other.input_count;
            param = std::move(other.param);
            input = std::move(other.input);
            output = std::move(other.output);
            m_log = std::move(other.m_log);
            return *this;
        }
        */

        enum class Status : int {
            OK,
            SKIP,
            WARNING,
            FAILED,
        };

        bool save(const std::string &root) {
            mkdir(root);
            assert(param_count == param.size());
            assert(input_count == input.size());
            assert(output_count == output.size());
            // write infos
            {
                std::ofstream fo(root + "/0." + op + ".txt");
                fo << param_count << std::endl;
                fo << input_count << std::endl;
                fo << output_count << std::endl;
            }
            {
                for (auto &pair : param) {
                    tensor::save(root + "/1." + pair.first + ".t", pair.second);
                }
            }
            {
                for (int i = 0; i < input_count; ++i) {
                    tensor::save(root + "/2.input_" + std::to_string(i) + ".t", input.at(i));
                }
            }
            {
                for (int i = 0; i < output_count; ++i) {
                    tensor::save(root + "/3.output_" + std::to_string(i) + ".t", output.at(i));
                }
            }
            return true;
        }

        Status convert_type(DTYPE convert_type, DeviceType device, bool strict) {
            bool support_flag = false;

            switch (convert_type)
            {
            case ts::FLOAT16:
                support_flag = (Fp16OperatorCreator::Query(device, op, strict) != nullptr);
                if (op == name::layer::cast() || op == name::layer::to_float())
                    support_flag = false;
                if (support_flag) {
                    //convert to fp16 if we can
                    for (auto &pair : param) {
                        auto type = pair.second.dtype();
                        if (type == FLOAT16 || type == FLOAT32 || type == FLOAT64) {
                            pair.second = tensor::cast(FLOAT16, pair.second);
                        }
                    }

                    for (int i = 0; i < input_count; ++i) {
                        auto type = input.at(i).dtype();
                        if (type == FLOAT16 || type == FLOAT32 || type == FLOAT64) {
                            input.at(i) = tensor::cast(FLOAT16, input.at(i));
                        }
                    }

                    for (int i = 0; i < output_count; ++i) {
                        auto type = output.at(i).dtype();
                        if (type == FLOAT16 || type == FLOAT32 || type == FLOAT64) {
                            output.at(i) = tensor::cast(FLOAT16, output.at(i));
                        }
                    }
                }
                break;
            default:
                TS_LOG_INFO << "don't support convert type: " << type_str(convert_type);
                return Status::FAILED;
            }  

            if (support_flag)
                return Status::OK;
            return Status::SKIP;
        }

        class SampleMatched {
        public:
            SampleMatched(const std::string &head, const std::string &tail, const std::string &body) {
                m_matched = {head, body, tail};
            }

            SampleMatched() : SampleMatched("", "", "") {}

            const std::string &str(size_t i) const {
                return m_matched[i];
            }

        private:
            std::vector<std::string> m_matched;
        };

        static bool simple_match(const std::string &str, SampleMatched &matched, const std::string &head, const std::string &tail) {
            if (str.length() < head.length() + tail.length()) return false;
            if (str.substr(0, head.length()) != head) return false;
            if (str.substr(str.length() - tail.length(), tail.length()) != tail) return false;
            auto body = str.substr(head.length(), str.length() - head.length() - tail.length());
            matched = SampleMatched(head, tail, body);
            return true;
        }

        // try load test case in files, throw exception if there is an broken case
        bool load(const std::string &root, const std::vector<std::string> &filenames) {
            TestCase tc;
            for (auto &filename : filenames) {
                if (filename.length() < 5 || filename[1] != '.') continue;

                auto fullpath = (root.empty() ? std::string() : root + "/");
                fullpath += filename;

                auto type_str = filename.substr(0, 1);
                auto type = std::strtol(type_str.c_str(), nullptr, 10);
                switch (type) {
                    default:
                        continue;
                    case 0: {
                        // std::regex pattern(R"(^0\.(.*)\.txt$)");
                        // std::smatch matched;
                        SampleMatched matched;
                        if (!simple_match(filename, matched, "0.", ".txt")) continue;
                        std::fstream ifile(fullpath);
                        if (!(ifile >> tc.param_count >> tc.input_count >> tc.output_count)) {
                            TS_LOG_ERROR << "format error in: " << fullpath << eject;
                            return false;
                        }
                        if (!tc.op.empty()) {
                            TS_LOG_ERROR << "Found two operator description in " << root << ": " << matched.str(1) << " vs. " << tc.op << eject;
                        }
                        tc.op = matched.str(1);
                        break;
                    }
                    case 1: {
                        // std::regex pattern(R"(^1\.(.*)\.t$)");
                        // std::smatch matched;
                        SampleMatched matched;
                        if (!simple_match(filename, matched, "1.", ".t")) continue;
                        FileStreamReader ifile(fullpath);
                        std::string name = matched.str(1);
                        Tensor value;
                        value.externalize(ifile);
                        tc.param.insert(std::make_pair(std::move(name), std::move(value)));
                        break;
                    }
                    case 2: {
                        // std::regex pattern(R"(^2\.input_(.*)\.t$)");
                        // std::smatch matched;
                        SampleMatched matched;
                        if (!simple_match(filename, matched, "2.input_", ".t")) continue;
                        FileStreamReader ifile(fullpath);
                        std::string id_str = matched.str(1);
                        auto id = int(std::strtol(id_str.c_str(), nullptr, 10));
                        Tensor value;
                        value.externalize(ifile);
                        tc.input.insert(std::make_pair(id, std::move(value)));
                        break;
                    }
                    case 3: {
                        // std::regex pattern(R"(^3\.output_(.*)\.t$)");
                        // std::smatch matched;
                        SampleMatched matched;
                        if (!simple_match(filename, matched, "3.output_", ".t")) continue;
                        FileStreamReader ifile(fullpath);
                        std::string id_str = matched.str(1);
                        auto id = int(std::strtol(id_str.c_str(), nullptr, 10));
                        Tensor value;
                        value.externalize(ifile);
                        tc.output.insert(std::make_pair(id, std::move(value)));
                        break;
                    }
                }
            }
            // not an test case
            if (tc.op.empty()) return false;

            // check format
            if (tc.param.size() != tc.param_count) {
                TS_LOG_ERROR << "Param count mismatch in " << root << ": "
                             << tc.param_count << " needed with " << tc.param.size() << " given." << eject;
            }
            if (tc.input.size() != tc.input_count) {
                TS_LOG_ERROR << "Input count mismatch in " << root << ": "
                             << tc.input_count << " needed with " << tc.input.size() << " given." << eject;
            }
            for (int i = 0; i < tc.input_count; ++i) {
                if (tc.input.find(i) == tc.input.end()) {
                    TS_LOG_ERROR << "Input missing in " << root << ": "
                                 << "1.input_" << i << ".t needed." << eject;
                }
            }
            if (tc.output.size() != tc.output_count) {
                TS_LOG_ERROR << "Output count mismatch in " << root << ": "
                             << tc.output_count << " needed with " << tc.output.size() << " given." << eject;
            }
            for (int i = 0; i < tc.output_count; ++i) {
                if (tc.output.find(i) == tc.output.end()) {
                    TS_LOG_ERROR << "Output missing in " << root << ": "
                                 << "1.output_" << i << ".t needed." << eject;
                }
            }

            TS_AUTO_ASSERT(tc.output_count == 1);

            *this = std::move(tc);

            // format succeed
            return true;
        }

        Status run(const ComputingDevice &device, int loop_count = 100) {
            m_log.str("");

            if (op.empty()) {
                m_log << "[ERROR]: " << "operator is empty." << std::endl;
                return Status::FAILED;
            }

            std::shared_ptr<Workbench> bench_ptr;
            try {
                bench_ptr = std::make_shared<Workbench>(device);   
            } catch (const Exception &e) {
                TS_LOG_ERROR << e.what();
                return Status::FAILED;
            }
    
            Workbench &bench = *bench_ptr;

            Bubble bubble(op, op, output_count);
            for (auto &param_pair: param) {
                bubble.set(param_pair.first, param_pair.second);
            }

            Operator::shared built_op;
            try {
                built_op = bench.offline_create(bubble, true);
            } catch (const OperatorNotFoundException &e) {
                m_log << "[SKIP]: " << "Not supported operator \"" << op << "\" for " << device << std::endl;
                return Status::SKIP;
            } catch (const Exception &e) {
                m_log << "[FAILED]: " << e.what() << std::endl;
                return Status::FAILED;
            }

            TS_AUTO_ASSERT(output_count == 1);

            std::vector<Tensor> input_vector(input_count);
            std::vector<Tensor> output_vector(output_count);

            for (auto &input_pair : input) {
                input_vector[input_pair.first] = input_pair.second;
            }

            for (auto &output_pair : output) {
                output_vector[output_pair.first] = output_pair.second;
            }

            std::vector<Tensor::Prototype> output_protos;
            bench.offline_infer(built_op, input_vector, output_protos);

            m_log << "Wanted: " << plot_line(input_vector, TensorPrototype(output_vector[0])) << std::endl;

            if (TensorPrototype(output_vector[0]) != TensorPrototype(output_protos)) {
                m_log << "Infer:  " << plot_line(input_vector, TensorPrototype(output_protos)) << std::endl;
                return Status::FAILED;
            }

            std::vector<Tensor> run_output;
            bench.offline_run(built_op, input_vector, run_output);

            TS_AUTO_ASSERT(run_output.size() == 1);

            if (TensorPrototype(output_vector[0]) != TensorPrototype(run_output[0])) {
                m_log << "Run:    " << plot_line(input_vector, TensorPrototype(run_output[0])) << std::endl;
                return Status::FAILED;
            }

            static const float MAX_MAX = 1e-4f;
            static const float MAX_AVG = 1e-5f;

            ctx::bind<DeviceContext> _bind_device_context(bench.device());
            // check diff
            Status succeed = Status::OK;
            float max, avg;
            auto fields_count = output_vector[0].fields_count();
            for (int i = 0; i < fields_count; ++i) {
                auto x = output_vector[0].field(i);
                auto y = run_output[0].field(i);
                diff(x, y, max, avg);
                if (max > MAX_MAX || avg > MAX_AVG)  {
                    if (max < MAX_MAX * 10 && avg < MAX_AVG * 10) {
                        m_log << "[WARNING] Diff output " << i << ": max = " << max << ", " << "avg = " << avg << std::endl;
                        if (succeed != Status::FAILED) succeed = Status::WARNING;
                    } else {
                        m_log << "[FAILED] Diff output " << i << ": max = " << max << ", " << "avg = " << avg << std::endl;
                        // TODO: log value
//                        plot_tensor(m_log, x) << std::endl;
//                        plot_tensor(m_log, y) << std::endl;
                        succeed = Status::FAILED;
                    }
                } else {
                    m_log << "[OK] Diff output " << i << ": max = " << max << ", " << "avg = " << avg << std::endl;
                }
            }

            if (succeed != Status::OK) return succeed;

            return Status::OK;
        }

        std::string log() { return m_log.str(); }

    private:
        std::ostringstream m_log;
    };
}

#endif //TENSORSTACK_TEST_CASE_HPP
