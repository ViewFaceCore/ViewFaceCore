//
// Created by Lby on 2017/7/31.
//

#ifndef ORZ_IO_CSV_H
#define ORZ_IO_CSV_H

#include <vector>
#include <string>

namespace orz {
    static const char CSVSeparator = ',';

    std::vector<std::vector<std::string> > CSVRead(const std::string &filename, char sep = CSVSeparator);

    void CSVWrite(const std::string &filename, const std::vector<std::vector<std::string> > &csv, char sep = CSVSeparator);

    std::string CSVBuild(const std::vector<std::string> &line, char sep = CSVSeparator);

    std::vector<std::string> CSVParse(const std::string &line, char sep = CSVSeparator);

    std::string CSVBuildItem(const std::string &item, char sep = CSVSeparator);
}

#endif //ORZ_IO_CSV_H
