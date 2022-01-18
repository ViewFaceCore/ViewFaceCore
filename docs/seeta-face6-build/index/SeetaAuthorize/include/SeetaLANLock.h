#ifndef INC_SEETA_LAN_LOCK_H
#define INC_SEETA_LAN_LOCK_H

#ifndef SEETA_LOCK_C_API
#ifdef _MSC_VER
#ifdef SEETA_LOCK_EXPORTS
#define SEETA_LOCK_API __declspec(dllexport)
#else
#define SEETA_LOCK_API __declspec(dllimport)
#endif
#else
#define SEETA_LOCK_API __attribute__ ((visibility("default")))
#endif

#define SEETA_LOCK_C_API extern "C" SEETA_LOCK_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

//SEETA_LOCK_C_API void SeetaLock_SetServerAddress(const char *ip, int port);
struct SeetaLock_Function;
SEETA_LOCK_C_API void SeetaLock_call(SeetaLock_Function *function);

#ifdef __cplusplus
}
#endif

#endif // INC_SEETA_LAN_LOCK_H
