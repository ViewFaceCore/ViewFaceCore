#pragma once

#include "seeta/Common/CStruct.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

struct SeetaLabeledBox
{
    SeetaRect pos;
    int label;
    float score;
};

struct SeetaLabeledBoxArray
{
    struct SeetaLabeledBox *data;
    size_t size;
};

#ifdef __cplusplus
}
#endif
