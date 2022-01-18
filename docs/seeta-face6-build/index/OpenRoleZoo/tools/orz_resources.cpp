//
// Created by kier on 2018/9/27.
//

#include <orz/tools/option.h>
#include <orz/tools/resources.h>
#include <orz/io/dir.h>
#include <orz/utils/format.h>
#include <orz/tools/cpp_resources.h>

void print_help(const orz::arg::OptionSet &options) {
    std::cout << "Usage: command [option] input_path" << std::endl;
    std::cout << "    " << "Input path can be file or folder" << std::endl;
    std::cout << "Option:" << std::endl;
    for (auto &option : options) {
        std::cout << "    " << option << std::endl;
    }
}

static bool is_abosulte_path(const std::string &path) {
    if (path.size() >= 1 && path[0] == '/') return true;
    if (path.size() >= 3 && path[1] == ':' && (path[2] == '/' || path[2] == '\\')) return true;
    return false;
}

int main(int argc, const char *argv[]) {
    orz::arg::OptionSet options;
    auto option_out_dir = options.add(orz::arg::STRING, {"o", "-out_dir"})->
            description("set generated files output dir")->
            value(orz::getcwd());
    auto option_in_dir = options.add(orz::arg::STRING, {"i", "-in_dir"})->
            description("set resources files input root dir")->
            value(orz::getcwd());
    auto option_filename = options.add(orz::arg::STRING, {"n", "-fn", "-filename"})->
            description("set output filename")->
            value("orz_resources");
    auto option_help = options.add(orz::arg::BOOLEAN, {"?", "h", "-help"})->
            description("print help documents");
    auto option_cpp = options.add(orz::arg::BOOLEAN, {"cpp", "-cpp"})->
            description("use cpp mode");
    std::string input_path;

    std::vector<std::string> args(argv + 1, argv + argc);

    if (!options.parse(args)) {
        std::cerr << options.last_error_message() << std::endl;
        return 1;
    }

    if (option_help->found()) {
        print_help(options);
        return 0;
    }

    if (!options.check()) {
        std::cerr << options.last_error_message() << std::endl;
        return 2;
    }

    if (args.size() < 1) {
        print_help(options);
        return 3;
    } else if (args.size() > 1) {
        std::cout << "[Info] Ignore parameters:";
        for (size_t i = 1; i < args.size(); ++i) {
            std::cout << " " << args[i];
        }
        std::cout << std::endl;
    }

    if (option_cpp->found()) {
        std::cout << "[Info] opetion --filename ignored." << std::endl;
        auto working_root = orz::getcwd();

        input_path = args[0];
        auto in_dir = option_in_dir->value().to_string();
        auto out_dir = option_out_dir->value().to_string();

        if (!option_in_dir->found()) {
            in_dir = orz::cut_path_tail(input_path);
        }

        orz::resources::CPPCompiler compiler;
        compiler.set_split(10);
        std::string header;
        std::vector<std::string> source;
        if (!compiler.compile(working_root, input_path, in_dir, out_dir, header, source)) {
            std::cerr << compiler.last_error_message() << std::endl;
            return 5;
        }
        return 0;
    }

    input_path = args[0];
    auto in_dir = option_in_dir->value().to_string();
    auto out_dir = option_out_dir->value().to_string();
    auto filename = option_filename->value().to_string();

    if (!option_in_dir->found()) {
        in_dir = orz::cut_path_tail(input_path);
    }

    orz::mkdir(out_dir);

    std::string header_filename = filename + ".h";
    std::string source_filename = filename + ".c";

    orz::resources::compiler compiler;

    compiler.set_input_directory(in_dir);
    compiler.set_output_directory(out_dir);

    if (is_abosulte_path(input_path)) {
        compiler.set_mark(input_path);
    } else {
        compiler.set_mark(orz::Join({orz::getcwd(), input_path}, orz::FileSeparator()));
    }

    if (!compiler.compile(input_path, header_filename, source_filename)) {
        std::cerr << compiler.last_error_message() << std::endl;
        return 4;
    }

    return 0;
}