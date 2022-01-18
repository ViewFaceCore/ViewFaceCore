#include "orz/io/dir.h"

#include "orz/utils/format.h"
#include "orz/utils/platform.h"
#include <cstdio>
#include <cstdlib>

#if ORZ_PLATFORM_OS_WINDOWS

#include <direct.h>
#include <io.h>

#define ACCESS ::_access
#define MKDIR(a) ::_mkdir((a))
#define GETCWD(buffer, length) ::_getcwd((buffer), (length))
#define CHDIR(path) ::_chdir(path)

#include <Windows.h>
#include <sys/stat.h>

#elif  ORZ_PLATFORM_OS_LINUX || ORZ_PLATFORM_OS_MAC || ORZ_PLATFORM_OS_IOS

#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <fstream>
#include <orz/io/dir.h>


#define ACCESS ::access
#define MKDIR(a) ::mkdir((a),0755)
#define GETCWD(buffer, length) ::getcwd((buffer), (length))
#define CHDIR(path) ::chdir(path)

#endif

namespace orz {

    const std::string FileSeparator() {
#if ORZ_PLATFORM_OS_WINDOWS
        return "\\";
#else
        return "/";
#endif
    }

    bool mkdir_core(const std::string &dir) {
        int miss = ACCESS(dir.c_str(), 0);
        if (miss) {
            int failed = MKDIR(dir.c_str());
            if (failed) {
                return false;
            }
        }
        return true;
    }

    bool mkdir(const std::string &dir) {
        auto path = orz::Split(dir, "\\/");
        for (size_t i = 1; i <= path.size(); ++i) {
            if (path[i - 1].empty()) continue;
            auto local_path = orz::Join(std::vector<std::string>(path.begin(), path.begin() + i), FileSeparator());
            if (!mkdir_core(local_path)) return false;
        }
        return true;
    }

    bool access(const std::string &path) {
        int miss = ACCESS(path.c_str(), 0);
        return !miss;
    }

    bool remove(const std::string &filename) {
        return std::remove(filename.c_str()) == 0;
    }

    bool rename(const std::string &oldname, const std::string &newname) {
        return std::rename(oldname.c_str(), newname.c_str()) == 0;
    }

    bool copy(const std::string &fromfile, const std::string &tofile, bool force) {
#if ORZ_PLATFORM_OS_WINDOWS
        return CopyFileA(fromfile.c_str(), tofile.c_str(), !force) != FALSE;
#elif ORZ_PLATFORM_OS_LINUX
        return std::system(orz::Concat(force ? "cp -f " : "cp ", fromfile, ' ', tofile).c_str()) == 0;
#else
        std::ifstream input(fromfile, std::ios::binary);
        std::ofstream output(tofile, std::ios::binary);
        output << input.rdbuf();
        return true;
#endif
    }

    std::string getcwd() {
        auto pwd = GETCWD(nullptr, 0);
        if (pwd == nullptr) return std::string();
        std::string pwd_str = pwd;
        free(pwd);
        return std::move(pwd_str);
    }

    std::string getself() {
#if ORZ_PLATFORM_OS_WINDOWS
        char exed[1024];
        auto exed_size = sizeof(exed) / sizeof(exed[0]);
        auto link_size = GetModuleFileNameA(nullptr, exed, DWORD(exed_size));

        if (link_size <= 0) return std::string();

        return std::string(exed, exed + link_size);
#else
        char exed[1024];
        auto exed_size = sizeof(exed) / sizeof(exed[0]);

        auto link_size = readlink("/proc/self/exe", exed, exed_size);

        if (link_size <= 0) return std::string();

        return std::string(exed, exed + link_size);
#endif
    }

    std::string getexed() {
        auto self = getself();
        return cut_path_tail(self);
    }

    bool cd(const std::string &path) {
        return CHDIR(path.c_str()) == 0;
    }

    std::string cut_path_tail(const std::string &path) {
        std::string tail;
        return cut_path_tail(path, tail);
    }

    std::string cut_path_tail(const std::string &path, std::string &tail) {
        auto win_sep_pos = path.rfind('\\');
        auto unix_sep_pos = path.rfind('/');
        auto sep_pos = win_sep_pos;
        if (sep_pos == std::string::npos) sep_pos = unix_sep_pos;
        else if (unix_sep_pos != std::string::npos && unix_sep_pos > sep_pos) sep_pos = unix_sep_pos;
        if (sep_pos == std::string::npos) {
            tail = path;
            return std::string();
        }
        tail = path.substr(sep_pos + 1);
        return path.substr(0, sep_pos);
    }

    std::string cut_name_ext(const std::string &name_ext, std::string &ext) {
        auto dot_pos = name_ext.rfind('.');
        auto sep_pos = dot_pos;
        if (sep_pos == std::string::npos) {
            ext = std::string();
            return name_ext;
        }
        ext = name_ext.substr(sep_pos + 1);
        return name_ext.substr(0, sep_pos);
    }

    bool isdir(const std::string &path) {
        struct stat buf;
        if (stat(path.c_str(), &buf)) return false;
        return bool((S_IFDIR & buf.st_mode) != 0);
    }

    bool isfile(const std::string &path) {
        struct stat buf;
        if (stat(path.c_str(), &buf)) return false;
        return bool((S_IFREG & buf.st_mode) != 0);
    }

    std::string join_path(const std::vector<std::string> &paths) {
        return Join(paths, FileSeparator());
    }
}

