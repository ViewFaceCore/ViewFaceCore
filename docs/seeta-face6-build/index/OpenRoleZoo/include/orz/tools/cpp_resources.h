//
// Created by kier on 2019/3/19.
//

#ifndef ORZ_TOOLS_CPP_RESOURCES_H
#define ORZ_TOOLS_CPP_RESOURCES_H

#include <string>
#include <vector>

namespace orz {
    namespace resources {
        class BinaryFileInfo {
        public:
            int line = 0;
            std::string url;
            std::string path;
            std::string root;
        };

        class OutputFileInfo {
        public:
            std::string path;
            std::string root;
        };

        class CPPCompiler {
        public:
            static const char annotation;
            static const char *const header_name;
            static const char *const header_ext;
            static const char *const source_name;
            static const char *const source_ext;

            void set_split(int split) { m_split = split; }

            bool compile(const std::string &working_root,
                         const std::string &resources_path,
                         const std::string &binary_files_root,
                         const std::string &output_root,
                         std::string &header, std::vector<std::string> &source);

            bool compile(const std::string &working_root,
                         const std::vector<BinaryFileInfo> &input_binaries,
                         const std::string &output_root,
                         std::string &header, std::vector<std::string> &source);

            const std::string &last_error_message() const {
                return m_last_error_message;
            }

        private:
            int m_split = 10;
            std::string m_last_error_message;
        };
    }
}

#endif //ORZ_TOOLS_CPP_RESOURCES_H
