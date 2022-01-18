//
// Created by kier on 2019/3/25.
//

#include "run_test/walker.hpp"
#include "run_test/option.hpp"
#include "run_test/test_case.hpp"

#include "double_test/double_it.hpp"

#include <utils/box.h>

int main(int argc, const char *argv[]) {
    using namespace ts;

    if (argc < 3) {
        std::cerr << "Usage: <command> input_path output_path" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    auto root = input_path;
    auto subdirs = FindFlodersRecursively(root);

    for (auto &subdir : subdirs) {
        auto case_root = Join({root, subdir}, FileSeparator());
        auto case_filenames = FindFiles(case_root);
        TestCase tc;
        try {
            if (!tc.load(case_root, case_filenames)) {
                continue;
            }
        } catch (const Exception &e) {
            continue;
        }
        // get next batch version
        TestCase tc2 = sister(tc);
        // concat test case
        TestCase tc3 = concat(tc2, tc);
        // write case to file
        tc3.save(Join({output_path, subdir}, FileSeparator()));
    }

    return 0;
}

