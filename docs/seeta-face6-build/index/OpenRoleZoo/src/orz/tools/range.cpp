//
// Created by lby on 2018/3/22.
//

#include "orz/tools/range.h"

#include <cfenv>

namespace orz {
    static int safe_ceil(double x) {
        int result;
        int save_round = std::fegetround();
        std::fesetround(FE_UPWARD);
        result = int(std::lrint(x));
        std::fesetround(save_round);
        return result;
    }

    std::vector<std::pair<int, int>> split_bins(int first, int second, int bins) {
        if (second <= first) return {};
        if (bins <= 1) return {{first, second}};
        auto step = safe_ceil((double(second) - double(first)) / bins);
        if (step < 1) step = 1;
        auto anchor = first;

        std::vector<std::pair<int, int>> result_bins;
        while (anchor + step < second) {
            result_bins.emplace_back(std::make_pair(anchor, anchor + step));
            anchor += step;
        }
        if (anchor < second) {
            result_bins.emplace_back(std::make_pair(anchor, second));
        }
        return result_bins;
    }

    static long safe_lceil(double x) {
        long result;
        int save_round = std::fegetround();
        std::fesetround(FE_UPWARD);
        result = long(std::lrint(x));
        std::fesetround(save_round);
        return result;
    }

    std::vector<std::pair<size_t, size_t>> lsplit_bins(size_t first, size_t second, size_t bins) {
        if (second <= first) return {};
        if (bins <= 1) return {{first, second}};
        auto step = safe_lceil((double(second) - double(first)) / bins);
        if (step < 1) step = 1;
        auto anchor = first;

        std::vector<std::pair<size_t, size_t >> result_bins;
        while (anchor + step < second) {
            result_bins.emplace_back(std::make_pair(anchor, anchor + step));
            anchor += step;
        }
        if (anchor < second) {
            result_bins.emplace_back(std::make_pair(anchor, second));
        }
        return result_bins;
    }
}
