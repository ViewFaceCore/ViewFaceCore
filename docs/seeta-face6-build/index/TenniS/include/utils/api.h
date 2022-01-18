//
// Created by Lby on 2017/8/7.
//

#ifndef TENSORSTACK_UTILS_API_H
#define TENSORSTACK_UTILS_API_H

#include "platform.h"

#define TS_EXTERN_C extern "C"

#if TS_PLATFORM_OS_WINDOWS
#   define TS_DLL_IMPORT __declspec(dllimport)
#   define TS_DLL_EXPORT __declspec(dllexport)
#   define TS_DLL_LOCAL
#else
#   if defined(__GNUC__) && __GNUC__ >= 4
#       define TS_DLL_IMPORT __attribute__((visibility("default")))
#       define TS_DLL_EXPORT __attribute__((visibility("default")))
#       define TS_DLL_LOCAL  __attribute__((visibility("hidden")))
#   else
#       define TS_DLL_IMPORT
#       define TS_DLL_EXPORT
#       define TS_DLL_LOCAL
#   endif
#endif

#define TS_API TS_DLL_EXPORT

#ifdef __cplusplus
#   define TS_C_API TS_EXTERN_C TS_API
#else
#   define TS_C_API TS_API
#endif

#ifndef TS_UNUSED
#   define TS_UNUSED(x) ((void)(x))
#endif

#ifdef TS_USE_DEBUG_API
#define TS_DEBUG_API TS_API
#else
#define TS_DEBUG_API
#endif

#endif //TENSORSTACK_UTILS_API_H
