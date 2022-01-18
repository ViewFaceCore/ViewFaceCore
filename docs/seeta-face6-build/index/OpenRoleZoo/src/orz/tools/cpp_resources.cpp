//
// Created by kier on 2019/3/19.
//

#include <orz/tools/cpp_resources.h>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <cfenv>

#include "orz/tools/cpp_resources.h"
#include "orz/io/dir.h"
#include "orz/io/walker.h"
#include "orz/utils/format.h"
#include "orz/utils/log.h"
#include "orz/mem/need.h"
#include "orz/io/i.h"
#include "orz/tools/range.h"
#include "orz/sync/shotgun.h"

#include "hash_table.h"

namespace orz {
    namespace resources {
        const char CPPCompiler::annotation = '#';
        const char *const CPPCompiler::header_name = "orz_cpp_resources";
        const char *const CPPCompiler::header_ext = ".h";
        const char *const CPPCompiler::source_name = "orz_cpp_resources";
        const char *const CPPCompiler::source_ext = ".cpp";

        static const char *const code_main_source_include[] = {
                "#include <cstring>",
                "#include <cstdio>",
                "#include <cstdint>",
        };

        static const char *const code_part_source_include[] = {
                "#include <cstdint>",
                "#include <cstddef>",
        };

        static const char *const code_main_resources_block_struct[] = {
                "struct orz_resources_block {",
                "    const void *data;",
                "    uint64_t size;",
                "};",
        };

        static const char *const code_main_resources_item_struct[] = {
                "struct orz_resources_item {",
                "    const char *url;",
                "    struct orz_resources_block *block;",
                "};",
        };

        static const char *block_name_header = "orz_cpp_resources_block_";
        static const char *block_size_header = "orz_cpp_resources_block_size_";

        // static const char *resources_name_header = "orz_cpp_resources_";

        static const char *struct_resources_block_name = "orz_resources_block";
        // static const char *struct_resources_item_name = "orz_resources_item";

        // static const char *object_resources_name = "orz_cpp_resources_table";

        static const char *item_menu_name_header = "orz_resources_menu_";

        static const char *const code_source_declare_ELFhash[] = {
                "static unsigned int ELFhash(const char *str) {",
                "    unsigned int hash = 0;",
                "    unsigned int x = 0;",
                "    while (*str) {",
                "        hash = (hash << 4) + *str;",
                "        if ((x = hash & 0xf0000000) != 0) {",
                "            hash ^= (x >> 24);",
                "            hash &= ~x;",
                "        }",
                "        str++;",
                "    }",
                "    return (hash & 0x7fffffff);",
                "}",
        };

        static const char *const code_source_declare_orz_resources_node[] = {
                "struct orz_resources_node {",
                "    const char *key;",
                "    unsigned int hash;",
                "    int next;",
                "",
                "    struct orz_resources_block *block;",
                "};",
        };

        static const char *const code_source_declare_orz_resources_table_head[] = {
                "static struct orz_resources_node orz_resources_table[] = {",
        };

        static const char *const code_source_declare_orz_resources_table_tail[] = {
                "};",
        };

        static const char *const code_source_declare_orz_resources_table_size[] = {
                "static const unsigned int orz_resources_table_size =",
                "    sizeof(orz_resources_table) / sizeof(orz_resources_table[0]);",
        };

        static const char *const code_source_declare_orz_resources_table_find[] = {
                "static struct orz_resources_node *orz_resources_table_find(const char *key)",
                "{",
                "    // if (orz_resources_table_size == 0) return NULL;",
                "    unsigned int hash = ELFhash(key);",
                "    unsigned int index = hash % orz_resources_table_size;",
                "    while (orz_resources_table[index].key) {",
                "        struct orz_resources_node *node = &orz_resources_table[index];",
                "        if (hash == node->hash && strcmp(key, node->key) == 0) {",
                "            return node;",
                "        }",
                "        {",
                "            int next_index = node->next;",
                "            if (next_index < 0) break;",
                "            index = (unsigned int)(next_index);",
                "        }",
                "    }",
                "    return NULL;",
                "}",
        };

        static const char *const code_source_declare_orz_resources_table_empty_find[] = {
                "static struct orz_resources_node *orz_resources_table_find(const char *key)",
                "{",
                "    return NULL;",
                "}",
        };

        static const char *const code_source_declare_orz_resources_get[] = {
                "static orz_resources _orz_build_buffer(orz_resources_block *block) {",
                "    // prepare size",
                "    uint64_t size = 0;",
                "    auto anchor = block;",
                "    while (anchor->data != nullptr) {",
                "        size += anchor->size;",
                "        anchor++;",
                "    }",
                "    orz_resources res((size_t)(size));",
                "    size = 0;",
                "    anchor = block;",
                "    while (anchor->data != nullptr) {",
                "        std::memcpy(res.data.get() + size, anchor->data, anchor->size);",
                "",
                "        size += anchor->size;",
                "        anchor++;",
                "    }",
                "",
                "    return std::move(res);",
                "}",
                "",
                "const struct orz_resources orz_resources_get(const std::string &url) {",
                "    auto c_url = url.c_str();",
                "    if (c_url[0] == '@')",
                "    {",
                "        c_url++;",
                "    }",
                "    struct orz_resources_node *node = orz_resources_table_find(c_url);",
                "    if (node == nullptr) return orz_resources();",
                "    return _orz_build_buffer(node->block);",
                "}",
        };

        static const char *const code_header_all[] = {
                "#ifndef _INC_ORZ_CPP_RESOURCES_AUTO_COMPILATION_H",
                "#define _INC_ORZ_CPP_RESOURCES_AUTO_COMPILATION_H",
                "",
                "",
                "#include <string>",
                "#include <memory>",
                "",
                "/**",
                " * \\brief ORZ resources structure",
                " */",
                "class orz_resources {",
                "public:",
                "    orz_resources() = default;",
                "    explicit orz_resources(size_t size) : size(size) {",
                "        data.reset(new char[size], std::default_delete<char[]>());",
                "    }",
                "",
                "    std::shared_ptr<char> data;    ///< memory pointer to the resource",
                "    size_t size = 0;               ///< size of the resource",
                "};",
                "",
                "#ifdef _MSC_VER",
                "#define ORZ_RESOURCES_HIDDEN_API",
                "#else",
                "#define ORZ_RESOURCES_HIDDEN_API __attribute__((visibility(\"hidden\")))",
                "#endif",
                "",
                "/**",
                " * \\brief Get ORZ resource by URL",
                " * \\param url The URL described in the orc file",
                " * \\return return \\c `struct orz_resources`",
                " * \\note Return { NULL, 0 } if failed.",
                " * \\note It will ignore the symbol `@` at the beginning of the string.",
                " */",
                "ORZ_RESOURCES_HIDDEN_API",
                "const orz_resources orz_resources_get(const std::string &url);",
                "",
                "",
                "#endif //_INC_ORZ_CPP_RESOURCES_AUTO_COMPILATION_H",
        };

        static inline std::ostream &write_lines(std::ostream &out, const char *const *lines, size_t num) {
            for (size_t i = 0; i < num; ++i) {
                out << lines[i] << std::endl;
            }
            return out;
        }

        template<size_t _Size>
        static inline std::ostream &write_lines(std::ostream &out, const char *const (&lines)[_Size]) {
            return write_lines(out, lines, _Size);
        }

        class cpp_working_in {
        public:
            using self = cpp_working_in;

            explicit cpp_working_in(const std::string &path, const std::string &base = "")
                    : m_backup(orz::getcwd()) {
                if (!base.empty()) orz::cd(base);
                if (!path.empty()) orz::cd(path);
                m_path = orz::getcwd();
            }

            ~cpp_working_in() {
                orz::cd(m_backup);
            }

            cpp_working_in(const self &) = delete;

            cpp_working_in &operator=(const self &) = delete;

            const std::string &path() const { return m_path; }

        private:
            std::string m_backup;
            std::string m_path;
        };

        static std::string trim(const std::string &line) {
            std::string pattern = " \r\n\t";
            auto left = line.find_first_not_of(pattern);
            if (left == std::string::npos) return "";
            auto right = line.find_last_not_of(pattern);
            if (right == std::string::npos) return "";
            if (right >= left) {
                return line.substr(left, right - left + 1);
            }
            return "";
        }

        static std::string to_url(const std::string &url) {
            auto fixed_url = url;
            for (auto &ch : fixed_url) {
                if (ch == '\\') ch = '/';
            }
            return std::move(fixed_url);
        }

        /**
         * Found binary_files, or return error message
         * @param working_root
         * @param resources_path
         * @param binary_files_root
         * @param binary_files
         * @return
         */
        static std::string find_binary_files(
                const std::string &working_root,
                const std::string &resources_path,
                const std::string &binary_files_root,
                std::vector<BinaryFileInfo> &binary_files) {
            cpp_working_in _(working_root);
            if (isdir(resources_path)) {
                auto files = FindFilesRecursively(resources_path);
                binary_files.clear();
                for (auto &file : files) {
                    BinaryFileInfo file_info;
                    file_info.url = "/" + to_url(file);
                    file_info.root = resources_path;
                    file_info.path = file;
                    binary_files.emplace_back(std::move(file_info));
                }
                std::cout << "[INFO] "
                          << "Found " << binary_files.size() << " files in folder \"" << resources_path << "\"."
                          << std::endl;
            } else {
                std::ifstream file_resources(resources_path);
                if (!file_resources.is_open()) return Concat("[ERROR] ", "Can not access \"", resources_path, "\"");
                binary_files.clear();
                int line_number = 0;
                std::string line;
                while (std::getline(file_resources, line)) {
                    line_number++;

                    line = trim(line);
                    if (line.empty()) continue;
                    if (line[0] == '#') continue;

                    auto split_index = line.find(':');

                    if (split_index == std::string::npos) {
                        return Concat("[ERROR] ", "line(", line_number, "): Syntax error: missing \':\'");
                    }

                    BinaryFileInfo file_info;
                    file_info.line = line_number;
                    file_info.url = trim(line.substr(0, split_index));
                    file_info.root = binary_files_root;
                    file_info.path = trim(line.substr(split_index + 1));

                    binary_files.emplace_back(std::move(file_info));
                }
                std::cout << "[INFO] "
                          << "Found " << binary_files.size() << " files in file \"" << resources_path << "\"."
                          << std::endl;
            }
            return "";
        }

        struct FileNo {
            std::ostream *header;
            std::ostream *source;
            std::vector<std::ofstream*> database;
        };

        static void close_filelist(const std::vector<std::ofstream*> &filelist) {
            for (auto &stream : filelist) {
                if (stream) stream->close();
            }
        }

        static std::string open_filelist(const std::string &root,
                const std::string &head, const std::string &tail, int count,
                std::vector<std::ofstream*> &filelist) {
            filelist.clear();
            filelist.resize(count);
            int serial_number = 0;
            try {
                for (auto &stream : filelist) {
                    std::string this_filename = Concat(head, ".", serial_number++, tail);
                    stream = new std::ofstream(this_filename);
                    if (!stream->is_open()) throw Concat("[ERROR] ", "Can not access \"", this_filename, "\" in ", root);
                }
            } catch (const std::string &message) {
                close_filelist(filelist);
                return message;
            } catch (...) {
                close_filelist(filelist);
                return "[ERROR] unknown failure.";
            }
            return "";
        }

        static void close_binary_files(const std::vector<std::ifstream*> &filelist) {
            for (auto &stream : filelist) {
                if (stream) stream->close();
            }
        }

        static std::string open_binary_files(const std::string &working_root,
                const std::vector<BinaryFileInfo> &binary_files,
                std::vector<std::ifstream*> &filelist) {
            filelist.clear();
            filelist.resize(binary_files.size());
            try {
                for (size_t i = 0; i < binary_files.size(); ++i) {
                    auto &bf = binary_files[i];
                    auto &stream = filelist[i];
                    cpp_working_in _(bf.root, working_root);
                    const std::string &this_filename = bf.path;
                    stream = new std::ifstream(this_filename, std::ios::binary);
                    if (!stream->is_open()) throw Concat("[ERROR] ", "Can not access \"", this_filename, "\" in ", bf.root);
                }
            } catch (const std::string &message) {
                close_binary_files(filelist);
                return message;
            } catch (...) {
                close_binary_files(filelist);
                return "[ERROR] unknown failure.";
            }
            return "";
        }

        static std::string memory_size_string(size_t memory_size) {
            static const char *base[] = {"B", "KB", "MB", "GB", "TB"};
            static const size_t base_size = sizeof(base) / sizeof(base[0]);
            auto number = double(memory_size);
            size_t base_time = 0;
            while (number >= 1024.0 && base_time + 1 < base_size) {
                number /= 1024.0;
                base_time++;
            }
            number = std::round(number * 10.0) / 10.0;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << number << base[base_time];
            return oss.str();
        }

        static size_t filesize(std::ifstream &in) {
            in.seekg(0, in.end);
            auto size = in.tellg();
            in.seekg(0, in.beg);
            return size;
        }

        static std::string write_header(FileNo &file_no) {
            auto &header = *file_no.header;
            write_lines(header, code_header_all) << std::endl;
            return "";
        }

        static long safe_lceil(double x) {
            long result;
            int save_round = std::fegetround();
            std::fesetround(FE_UPWARD);
            result = long(std::lrint(x));
            std::fesetround(save_round);
            return result;
        }

        class Schedule {
        public:
            class Task {
            public:
                std::ifstream *in;  // in binary file
                std::ofstream *out; // out text source file
                size_t count;   // how many bytes to write
            };
            std::vector<Task> list; // a list to write binary to sources files
            std::map<std::ofstream *, size_t> size;   // the file ready to write
            std::map<std::ifstream *, std::vector<size_t>> menu;   // in file stream to witch part to list

            void write(std::ifstream *in, size_t count, std::vector<std::ofstream*> &out) {
                auto bins = split(count, out.size());
                auto out_streams = find_smallest(out, int(bins.size()));
                for (size_t i = 0; i < bins.size(); ++i) {
                    auto &bin = bins[i];
                    auto &stream = out_streams[i];

                    Schedule::Task task;
                    task.in  = in;
                    task.out = stream;
                    task.count = bin.second - bin.first;

                    this->size[stream] += task.count;
                    this->menu[in].emplace_back(this->list.size());
                    this->list.emplace_back(task);
                }
            }

        private:
            static std::vector<std::pair<size_t, size_t>> split(size_t count, size_t bins) {
                auto count_8 = size_t(safe_lceil(double(count) / 8.0));
                auto contant = lsplit_bins(0, count_8, bins);
                for (auto &bin : contant) {
                    bin.first *= 8;
                    bin.second *= 8;
                }
                contant.back().second = count;
                return contant;
            }

            std::vector<std::ofstream *> find_smallest(std::vector<std::ofstream*> &out, int top_n) {
                using size_item = std::pair<std::ofstream*,size_t>;
                std::vector<size_item> file_size;
                for (auto &f : out) {
                    auto it = this->size.find(f);
                    if (it == this->size.end()) {
                        auto n = this->size.insert(std::make_pair(f, 0));
                        it = n.first;
                    }
                    file_size.emplace_back(std::move(*it));
                }
                std::partial_sort(file_size.begin(), file_size.begin() + top_n, file_size.end(),
                                  [](const size_item &lhs, const size_item &rhs) {
                                      return lhs.second < rhs.second;
                                  });
                std::vector<std::ofstream *> smallest;
                for (auto it = file_size.begin(); it != file_size.begin() + top_n; ++it) {
                    smallest.emplace_back(it->first);
                }
                return smallest;
            }
        };

        static void write_main_source_file(
                Schedule &schedule, std::ostream &source,
                const std::vector<BinaryFileInfo> &fileinfo,
                const std::vector<std::ifstream *> &filestream) {
            source << "#include \"" << CPPCompiler::header_name << CPPCompiler::header_ext << "\"" << std::endl << std::endl;

            write_lines(source, code_main_source_include) << std::endl;

            write_lines(source, code_main_resources_block_struct) << std::endl;
            write_lines(source, code_main_resources_item_struct) << std::endl;

            for (size_t i = 0; i < schedule.list.size(); ++i) {
                // auto &task = schedule.list[i];
                source << "extern const uint64_t " << block_name_header << i << "[];" << std::endl;
                source << "extern const uint64_t " << block_size_header << i << ";" << std::endl;
                source << std::endl;
            }

            for (int i = 0; i < fileinfo.size(); ++i) {
                source << "static " << struct_resources_block_name << " "
                 << item_menu_name_header << i << "[] = {" << std::endl;

                // auto &info = fileinfo[i];
                auto &stream = filestream[i];

                auto &list = schedule.menu[stream];
                for (auto j : list) {
                    source << "    {" << block_name_header << j << ", " << block_size_header << j << "}," << std::endl;
                }
                source << "    {nullptr, 0}, " << std::endl;

                source << "};" << std::endl;
                source << std::endl;
            }

            hash_table<int> table;
            for (size_t i = 0; i < fileinfo.size(); ++i) {
                auto &info = fileinfo[i];
                table.insert(info.url, i);
            }
            auto &nodes = table.nodes();

            // write hush table
            write_lines(source, code_source_declare_ELFhash) << std::endl;
            write_lines(source, code_source_declare_orz_resources_node) << std::endl;
            write_lines(source, code_source_declare_orz_resources_table_head);

            for (auto &node : nodes) {
                if (node == nullptr) {
                    source << "    {nullptr, 0, -1, nullptr, },";
                }
                else {
                    source << "    {\"" << node->key << "\", "
                           << node->hash << ", "
                           << node->next << ", "
                           << item_menu_name_header << node->value << ","
                           << "},";
                }
                source << std::endl;
            }

            write_lines(source, code_source_declare_orz_resources_table_tail) << std::endl;
            write_lines(source, code_source_declare_orz_resources_table_size) << std::endl;

            if (nodes.empty()) {
                write_lines(source, code_source_declare_orz_resources_table_empty_find) << std::endl;
            } else {
                write_lines(source, code_source_declare_orz_resources_table_find) << std::endl;
            }

            write_lines(source, code_source_declare_orz_resources_get) << std::endl;
        }

        static void write_part_block(const char *buffer, size_t count, int id, std::ofstream &source) {
            union uint64_chunk {
                uint64_t i;
                uint8_t c[8];

                uint64_chunk() : i(0) {}

                void zeros() { i = 0; }
            };
            uint64_chunk chunk;
            std::ostringstream source_buffer;
            static const int loop_size = 96;

            source << "extern const uint64_t " << block_size_header << id << " = " << count << "UL;" << std::endl;
            source << "extern const uint64_t " << block_name_header << id << "[] = {" << std::endl;

            size_t i = 0;
            if (count > 7) {
                for (i = 0; i < count - 7; i += 8) {
                    if (source_buffer.tellp() > loop_size) {
                        source_buffer << std::endl;
                        source << source_buffer.str();
                        source_buffer.str("");
                    }

                    std::memcpy(chunk.c, buffer + i, 8);
                    source_buffer << "0x" << std::hex << chunk.i << ",";
                }
            }

            if (source_buffer.tellp() > loop_size) {
                source_buffer << std::endl;
                source << source_buffer.str();
                source_buffer.str("");
            }

            if (i < count) {
                chunk.zeros();
                std::memcpy(chunk.c, buffer + i, size_t(count - i));
                source_buffer << "0x" << std::hex << chunk.i << ",";
            }

            if (source_buffer.tellp() > 0) {
                source_buffer << std::endl;
                source << source_buffer.str();
                source_buffer.str("");
            }
            source << "};" << std::endl;
        }

        static void write_part_source_file(Schedule &schedule, std::ifstream &in, size_t count, Shotgun &gun) {
            auto menu = schedule.menu[&in];

            std::unique_ptr<char[]> buffer(new char[count]);
            in.read(buffer.get(), count);

            size_t anchor = 0;

            for (size_t i : menu) {
                auto &task = schedule.list[i];
                auto &source = *task.out;

                auto begin = anchor;
                auto end = begin + task.count;
                anchor = end;

                gun.fire([&, begin, end, i](int) {
                    write_part_block(buffer.get() + begin, end - begin, i, source);
                });
            }
            gun.join();
        }

        static std::string write_source(FileNo &file_no,
                const std::vector<BinaryFileInfo> &fileinfo,
                const std::vector<std::ifstream *> &filestream) {
            Schedule schedule;
            for (size_t i = 0; i < fileinfo.size(); ++i) {
                // auto &info = fileinfo[i];
                auto &stream = *filestream[i];
                auto data_size = filesize(stream);
                schedule.write(&stream, data_size, file_no.database);
            }
            write_main_source_file(schedule, *file_no.source, fileinfo, filestream);

            // write header in part source files
            for (auto &part : file_no.database) {
                write_lines(*part, code_part_source_include) << std::endl;
            }

            Shotgun gun(schedule.size.size());

            for (size_t i = 0; i < fileinfo.size(); ++i) {
                auto &info = fileinfo[i];
                auto &stream = *filestream[i];
                auto data_size = filesize(stream);
                std::cout << "[Info] Compiling \"" << info.url << "\". " << memory_size_string(data_size) << std::endl;
                write_part_source_file(schedule, stream, data_size, gun);
            }

            return "";
        }


#define RETURN_WITH_ERROR(message) { m_last_error_message = (message); return false; }
#define CHECK_MESSAGE(condition) { if (!(m_last_error_message = (condition)).empty()) return false; }

        bool CPPCompiler::compile(const std::string &working_root, const std::vector<BinaryFileInfo> &input_binaries,
                                  const std::string &output_root,
                                  std::string &header, std::vector<std::string> &source) {
            auto local_working_root = working_root;
            if (local_working_root.empty()) local_working_root = getcwd();
            auto local_output_root = output_root;
            if (local_output_root.empty()) local_output_root = getcwd();

            if (!isdir(local_working_root)) {
                RETURN_WITH_ERROR(Concat("[ERROR] Can not change working dir to ", local_working_root));
            }

            // do in local_working_root and local_output_root
            {
                cpp_working_in _root(local_working_root);
                mkdir(local_output_root);
                cpp_working_in _(local_output_root);

                auto header_filename = Concat(header_name, header_ext);
                auto source_filename = Concat(source_name, source_ext);

                FileNo file_no;
                std::ostringstream header_in_memory;
                std::ofstream source_in_file(source_filename);
                if (!source_in_file.is_open())
                    RETURN_WITH_ERROR(Concat("[ERROR] ", "Can not access \"", source_filename, "\" in ", local_output_root));
                file_no.header = &header_in_memory;
                file_no.source = &source_in_file;
                CHECK_MESSAGE(open_filelist(local_output_root, source_name, source_ext, m_split, file_no.database));
                need _close_output(close_filelist, std::ref(file_no.database));

                std::vector<std::ifstream *> binary_stream_list;
                CHECK_MESSAGE(open_binary_files(local_working_root, input_binaries, binary_stream_list));
                need _close_input(close_binary_files, std::ref(binary_stream_list));

                // write header
                CHECK_MESSAGE(write_header(file_no));

                // check header
                std::string header_ready_contant = read_txt_file(header_filename);
                std::string header_contant = header_in_memory.str();
                if (header_contant != header_ready_contant) {
                    std::ofstream header_in_file(header_filename);
                    header_in_file << header_contant;
                }

                // write files from binary files
                CHECK_MESSAGE(write_source(file_no, input_binaries, binary_stream_list));
            }

            return true;
        }

        bool CPPCompiler::compile(const std::string &working_root, const std::string &resources_path,
                                  const std::string &binary_files_root, const std::string &output_root,
                                  std::string &header,
                                  std::vector<std::string> &source) {
            auto local_working_root = working_root;
            if (local_working_root.empty()) local_working_root = getcwd();
            auto local_binary_files_root = binary_files_root;
            if (local_binary_files_root.empty()) local_binary_files_root = getcwd();
            auto local_output_root = output_root;
            if (local_output_root.empty()) local_output_root = getcwd();

            if (!isdir(local_working_root)) {
                RETURN_WITH_ERROR(Concat("[ERROR] Can not change working dir to ", local_working_root));
            }

            std::vector<BinaryFileInfo> binary_files;
            CHECK_MESSAGE(find_binary_files(local_working_root, resources_path, local_binary_files_root, binary_files))
            return compile(local_working_root, binary_files, local_output_root, header, source);
        }
    }
}