#include "orz/io/walker.h"
#include <iostream>
#include <queue>

#include "orz/io/dir.h"
#include "orz/utils/platform.h"

#if ORZ_PLATFORM_OS_WINDOWS
#include <io.h>
#else
#include <dirent.h>
#include <cstring>
#endif

namespace orz {

    static std::vector<std::string> FindFilesCore(const std::string &path, std::vector<std::string> *dirs = nullptr) {
        std::vector<std::string> result;
        if (dirs) dirs->clear();
#if ORZ_PLATFORM_OS_WINDOWS
        _finddata_t file;
        std::string pattern = path + FileSeparator() + "*";
        auto handle = _findfirst(pattern.c_str(), &file);

        if (handle == -1L) return result;
        do {
            if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0) continue;
            if (file.attrib & _A_SUBDIR) {
                if (dirs) dirs->push_back(file.name);
            } else {
                result.push_back(file.name);
            }
        } while (_findnext(handle, &file) == 0);

        _findclose(handle);
#else
        struct dirent *file;

        auto handle = opendir(path.c_str());

        if (handle == nullptr) return result;

        while ((file = readdir(handle)) != nullptr)
        {
            if (strcmp(file->d_name, ".") == 0 || strcmp(file->d_name, "..") == 0) continue;
            if (file->d_type & DT_DIR)
            {
                if (dirs) dirs->push_back(file->d_name);
            }
            else if (file->d_type & DT_REG)
            {
                result.push_back(file->d_name);
            }
            // DT_LNK // for linkfiles
        }

        closedir(handle);
#endif
        return std::move(result);
    }

    std::vector<std::string> FindFiles(const std::string &path) {
        return FindFilesCore(path);
    }

    std::vector<std::string> FindFiles(const std::string &path, std::vector<std::string> &dirs) {
        return FindFilesCore(path, &dirs);
    }

    std::vector<std::string> FindFilesRecursively(const std::string &path, int depth) {
        std::vector<std::string> result;
        std::queue<std::pair<std::string, int> > work;
        std::vector<std::string> dirs;
        std::vector<std::string> files = FindFiles(path, dirs);
        result.insert(result.end(), files.begin(), files.end());
        for (auto &dir : dirs) work.push({dir, 1});
        while (!work.empty()) {
            auto local_pair = work.front();
            work.pop();
            auto local_path = local_pair.first;
            auto local_depth = local_pair.second;
            if (depth > 0 && local_depth >= depth) continue;
            files = FindFiles(path + FileSeparator() + local_path, dirs);
            for (auto &file : files) result.push_back(local_path + FileSeparator() + file);
            for (auto &dir : dirs) work.push({local_path + FileSeparator() + dir, local_depth + 1});
        }
        return result;
    }

    std::vector<std::string> FindFlodersRecursively(const std::string &path, int depth) {
        std::vector<std::string> result;
        std::queue<std::pair<std::string, int> > work;
        std::vector<std::string> dirs;
        std::vector<std::string> files = FindFiles(path, dirs);
        result.insert(result.end(), dirs.begin(), dirs.end());
        for (auto &dir : dirs) work.push({dir, 1});
        while (!work.empty()) {
            auto local_pair = work.front();
            work.pop();
            auto local_path = local_pair.first;
            auto local_depth = local_pair.second;
            if (depth > 0 && local_depth >= depth) continue;
            files = FindFiles(path + FileSeparator() + local_path, dirs);
            for (auto &dir : dirs) result.push_back(local_path + FileSeparator() + dir);
            for (auto &dir : dirs) work.push({local_path + FileSeparator() + dir, local_depth + 1});
        }
        return result;
    }
}
