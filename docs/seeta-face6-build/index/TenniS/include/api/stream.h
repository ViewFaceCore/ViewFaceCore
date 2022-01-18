//
// Created by kier on 19-4-25.
//

#ifndef TENNIS_API_STREAM_H
#define TENNIS_API_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/**
 * Callback function of stream writing.
 * @param obj stream object
 * @param data buffer to write
 * @param length length of buffer
 * @return actually write size
 */
typedef uint64_t ts_stream_write(void *obj, const char *data, uint64_t length);

/**
 * Callback function of stream reading.
 * @param obj stream object
 * @param data buffer to read
 * @param length length of buffer
 * @return actually read size
 */
typedef uint64_t ts_stream_read(void *obj, char *data, uint64_t length);

#ifdef __cplusplus
}
#endif

#endif //TENNIS_API_STREAM_H
