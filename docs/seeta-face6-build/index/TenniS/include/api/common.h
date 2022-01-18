//
// Created by keir on 2019/3/16.
//

#ifndef TENNIS_API_COMMON_H
#define TENNIS_API_COMMON_H

#define TENNIS_EXTERN_C extern "C"

#if defined(_MSC_VER)
#   define TENNIS_DLL_IMPORT __declspec(dllimport)
#   define TENNIS_DLL_EXPORT __declspec(dllexport)
#   define TENNIS_DLL_LOCAL
#else
#   if defined(__GNUC__) && __GNUC__ >= 4
#       define TENNIS_DLL_IMPORT __attribute__((visibility("default")))
#       define TENNIS_DLL_EXPORT __attribute__((visibility("default")))
#       define TENNIS_DLL_LOCAL  __attribute__((visibility("hidden")))
#   else
#       define TENNIS_DLL_IMPORT
#       define TENNIS_DLL_EXPORT
#       define TENNIS_DLL_LOCAL
#   endif
#endif

#if defined(BUILDING_TENNIS)
#define TENNIS_API TENNIS_DLL_EXPORT
#else
#define TENNIS_API TENNIS_DLL_IMPORT
#endif

#ifdef __cplusplus
#   define TENNIS_C_API TENNIS_EXTERN_C TENNIS_API
#else
#   define TENNIS_C_API TENNIS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef int32_t ts_bool;

/**
 * Get last error message in current thread.
 * @return last error message.
 * @note Return NULL if failed.
 */
TENNIS_C_API const char *ts_last_error_message();

/**
 * Set error message in current thread.
 * @param message error message.
 */
TENNIS_C_API void ts_set_error_message(const char *message);

#ifdef __cplusplus
}
#endif

#endif //TENNIS_API_COMMON_H
