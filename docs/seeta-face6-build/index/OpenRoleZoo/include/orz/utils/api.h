//
// Created by Lby on 2017/8/7.
//

#ifndef ORZ_UTILS_API_H
#define ORZ_UTILS_API_H

#include "platform.h"

#define ORZ_EXTERN_C extern "C"

#if ORZ_PLATFORM_OS_WINDOWS
#   define ORZ_DLL_IMPORT __declspec(dllimport)
#   define ORZ_DLL_EXPORT __declspec(dllexport)
#   define ORZ_DLL_LOCAL
#else
#   if defined(__GNUC__) && __GNUC__ >= 4
#       define ORZ_DLL_IMPORT __attribute__((visibility("default")))
#       define ORZ_DLL_EXPORT __attribute__((visibility("default")))
#       define ORZ_DLL_LOCAL  __attribute__((visibility("hidden")))
#   else
#       define ORZ_DLL_IMPORT
#       define ORZ_DLL_EXPORT
#       define ORZ_DLL_LOCAL
#   endif
#endif

#define ORZ_API ORZ_DLL_EXPORT

#ifdef __cplusplus
#   define ORZ_C_API ORZ_EXTERN_C ORZ_API
#else
#   define ORZ_C_API ORZ_API
#endif

#ifndef ORZ_UNUSED
#   define ORZ_UNUSED(x) ((void)x)
#endif

#endif //ORZ_UTILS_API_H
