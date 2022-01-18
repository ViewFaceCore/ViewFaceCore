//
// Created by kier on 2019/3/16.
//

#include <board/dump.h>

#include "board/dump.h"
#include "board/hook.h"
#include "core/tensor_builder.h"

#include <fstream>

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

    static const std::string FileSeparator() {
#if TS_PLATFORM_OS_WINDOWS
        return "\\";
#else
        return "/";
#endif
    }


    static bool mkdir_core(const std::string &dir) {
        int miss = ACCESS(dir.c_str(), 0);
        if (miss) {
            int failed = MKDIR(dir.c_str());
            if (failed) {
                return false;
            }
        }
        return true;
    }

    static bool mkdir(const std::string &dir) {
        auto path = Split(dir, "\\/");
        for (size_t i = 1; i <= path.size(); ++i) {
            if (path[i - 1].empty()) continue;
            auto local_path = Join(std::vector<std::string>(path.begin(), path.begin() + i), FileSeparator());
            if (!mkdir_core(local_path)) return false;
        }
        return true;
    }

    std::vector<Tensor> track(const std::string &dump_root, Workbench::shared bench, const std::vector<Tensor> &input) {
        Hook hook;
        ctx::bind<Hook> _bind_hook(hook);

        class BuildTestCase {
        public:
            const Operator *op;
            std::vector<Tensor> inputs;
            std::vector<Tensor> outputs;
        };

        std::vector<BuildTestCase> building;

        hook.before_run([&](const Hook::StructBeforeRun &info) {
            BuildTestCase btc;
            btc.op = info.op;
            for (int i = 0; i < info.stack->size(); ++i) {
                btc.inputs.push_back(info.stack->index(i)->clone());
            }
            building.push_back(std::move(btc));
        });

        hook.after_run([&](const Hook::StructAfterRun &info) {
            auto &btc = building.back();
            TS_AUTO_CHECK(info.op == btc.op);
            for (int i = 0; i < info.stack->size(); ++i) {
                btc.outputs.push_back(info.stack->index(i)->clone());
            }
        });

        for (int i = 0; i < input.size(); ++i) {
            bench->input(i, input[i]);
        }

        bench->run();

        std::vector<Tensor> outputs(size_t(bench->output_count()));
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i] = bench->output(int(i));
        }

        for (size_t i = 0; i < building.size(); ++i) {
            auto &btc = building[i];
            std::string subdir = dump_root + "/" + std::to_string(i) + "_" + btc.op->op() + "/";

            if (!mkdir(subdir)) {
                TS_LOG_INFO << "Can not access: " << subdir;
                continue;
            }

            TS_LOG_INFO << "Tracked: " << subdir;
            auto params = btc.op->params();

            std::string file_0 = subdir + "0." + btc.op->op() + ".txt";
            std::ofstream ofile(file_0);
            ofile << params.size() << std::endl;
            ofile << btc.inputs.size() << std::endl;
            ofile << btc.outputs.size() << std::endl;
            ofile.close();

            for (auto &param : params) {
                std::string file_1 = subdir + "1." + param.first + ".t";
                tensor::save(file_1, param.second);
            }

            for (int j = 0; j < btc.inputs.size(); ++j) {
                std::string file_2 = subdir + "2.input_" + std::to_string(j) + ".t";
                tensor::save(file_2, btc.inputs[j]);
            }

            for (int j = 0; j < btc.outputs.size(); ++j) {
                std::string file_3 = subdir + "3.output_" + std::to_string(j) + ".t";
                tensor::save(file_3, btc.outputs[j]);
            }
        }

        return std::move(outputs);
    }
}
