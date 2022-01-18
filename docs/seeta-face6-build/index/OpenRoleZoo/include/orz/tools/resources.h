//
// Created by kier on 2018/9/25.
//

#ifndef ORZ_TOOLS_RESOURCES_H
#define ORZ_TOOLS_RESOURCES_H

#include <string>
#include <vector>
#include <istream>
#include <ostream>

namespace orz {

    namespace resources {
        struct resources {
        public:
            resources() = default;
            explicit resources(const std::string &url, const std::string &path, size_t line = 0)
                    : line(line), url(url), path(path) {}

            size_t line = 0;
            std::string url;
            std::string path;
        };

        class compiler {
        public:
            static const char annotation = '#';

            compiler();

            bool compile(const std::vector<resources> &in_resources, std::ostream &out_header, std::ostream &out_source,
                         const std::string &val_header_path = "orz_resources.h");

            bool compile(std::istream &in_source, std::ostream &out_header, std::ostream &out_source,
                         const std::string &val_header_path = "orz_resources.h");

            bool compile(std::istream &in_source,
                         const std::string &header_filename,
                         const std::string &source_filename);

            bool compile(const std::string &path, const std::string &header_filename, const std::string &source_filename);

            bool up2date_header(const std::string &header_filename);

            bool up2date_header(const std::string &header_filename, const std::string &content);

            const std::string &last_error_message() const {
                return m_last_error_message;
            }

            void set_working_directory(const std::string &path) { m_working_directory = path; }
            void set_output_directory(const std::string &path) { m_output_directory = path; }
            void set_input_directory(const std::string &path) { m_input_directory = path; }
            void set_mark(const std::string &mark) { m_mark = mark; }

        private:
            std::string m_last_error_message;

            // path to load orc file
            std::string m_working_directory;

            // path to output generated files
            std::string m_output_directory;

            // path to input resources file
            std::string m_input_directory;

            std::string m_mark;
        };
    }
}


#endif //ORZ_TOOLS_RESOURCES_H
