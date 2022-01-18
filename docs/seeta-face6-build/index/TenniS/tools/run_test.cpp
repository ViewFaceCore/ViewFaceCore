//
// Created by kier on 2019/3/4.
//

#include "run_test/walker.hpp"
#include "run_test/option.hpp"
#include "run_test/test_case.hpp"

#include <utils/box.h>

int main(int argc, const char *argv[]) {
    using namespace ts;

    if (argc < 2) {
        std::cerr << "Usage: <command> path [device [id]]" << std::endl;
        return 1;
    }
    std::string root = argv[1];

    std::string device = "cpu";
    int id = 0;

    if (argc > 2) {
        device = argv[2];
    }

    for (auto &ch : device) {
        ch = char(std::tolower(ch));
    }

    if (argc > 3) {
        id = int(std::strtol(argv[3], nullptr, 10));
    }

    ComputingDevice computing_device(device, id);

    auto subdirs = FindFlodersRecursively(root);
    subdirs.emplace_back(".");

    int ok_count = 0;
    int skip_count = 0;
    int warning_count = 0;
    int failed_count = 0;

    for (auto &subdir : subdirs) {
        auto case_root = Join({root, subdir}, FileSeparator());
        auto case_filenames = FindFiles(case_root);
        TestCase tc;
        try {
            if (!tc.load(case_root, case_filenames)) {
                continue;
            }
        } catch (const Exception &e) {
            std::cerr << e.what() << std::endl;
        }
        // try infer
        TestCase::Status status = TestCase::Status::FAILED;
        try {
            status = tc.run(computing_device, 1);
        } catch (const Exception &e) {
            std::cerr << e.what() << std::endl;
        }
        // show log
        switch (status) {
            case TestCase::Status::OK: {
                ok_count++;
                std::cout << "[OK]: " << subdir << " on " << computing_device << std::endl;
                // std::cout << tc.log();
                break;
            }
            case TestCase::Status::SKIP: {
                skip_count++;
                std::cout << "[SKIP]: " << subdir << " on " << computing_device << std::endl;
                // std::cout << tc.log();
                break;
            }
            case TestCase::Status::WARNING: {
                warning_count++;
                std::cout << "-------------------------------------------------------------------------" << std::endl;
                std::cout << "[WARNING]: " << subdir << " on " << computing_device << std::endl;
                std::cout << tc.log();
                std::cout << "-------------------------------------------------------------------------" << std::endl;
                break;
            }
            case TestCase::Status::FAILED: {
                failed_count++;
                std::cout << "-------------------------------------------------------------------------" << std::endl;
                std::cout << "[FAILED]: " << subdir << " on " << computing_device << std::endl;
                std::cout << tc.log();
                std::cout << "-------------------------------------------------------------------------" << std::endl;
                break;
            }
        }
    }

    TS_LOG_INFO << "[OK]: " << ok_count <<
                ", [SKIP]: " << skip_count <<
                ", [WARNING]: " << warning_count <<
                ", [FAILED]: " << failed_count;

    return 0;
}
