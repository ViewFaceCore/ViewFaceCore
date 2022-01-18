//
// Created by Lby on 2017/7/31.
//

#include "orz/io/csv.h"

#include <regex>
#include <sstream>
#include <fstream>

namespace orz {
    std::vector<std::vector<std::string> > CSVRead(const std::string &filename, char sep) {
        std::ifstream file(filename);

        std::vector<std::vector<std::string> > csv;
        std::string line;
        while (std::getline(file, line)) {
            csv.push_back(CSVParse(line, sep));
        }

        file.close();

        return csv;
    }

    void CSVWrite(const std::string &filename, const std::vector<std::vector<std::string> > &csv, char sep) {
        std::ofstream file(filename);

        for (auto &line : csv) {
            file << CSVBuild(line, sep) << std::endl;
        }

        file.close();
    }

    std::string CSVBuild(const std::vector<std::string> &line, char sep) {
        std::ostringstream oss;
        for (size_t i = 0; i < line.size(); ++i) {
            if (i) oss << sep;
            oss << CSVBuildItem(line[i], sep);
        }
        return oss.str();
    }

    std::vector<std::string> CSVParse(const std::string &line, char sep) {
        bool have_quota = false;
        std::string item;
        const char *anchor = line.c_str();
        const char quota = '"';
        std::vector<std::string> result;
        while (true) {
            if (*anchor == '\0') {
                result.push_back(item);
                break;
            } else if (*anchor == sep) {
                if (!have_quota) {
                    result.push_back(item);
                    item.clear();
                } else {
                    item.push_back(*anchor);
                }
            } else if (*anchor == quota) {
                if (!have_quota) {
                    have_quota = true;
                } else {
                    if (anchor[1] == quota) {
                        item.push_back(quota);
                        ++anchor;
                    } else {
                        have_quota = false;
                    }
                }
            } else {
                item.push_back(*anchor);
            }
            ++anchor;
        }
        return result;
    }

    std::string CSVBuildItem(const std::string &item, char sep) {
        bool need_quota = false;
        if (item.find(sep) != std::string::npos) {
            need_quota = true;
        }
        std::regex rgx(R"(")");
        std::string fmt(R"("")");
        std::string build_item = std::regex_replace(item, rgx, fmt);
        if (build_item.length() != item.length()) {
            need_quota = true;
        }
        std::string quota = R"(")";
        if (need_quota) {
            build_item = quota + build_item + quota;
        }
        return build_item;
    }
}