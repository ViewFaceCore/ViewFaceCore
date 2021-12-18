#define STDCALL __attribute__((stdcall))
#define View_Api extern "C"

typedef void(STDCALL* LogCallBack)(const char* logText);
extern LogCallBack logger;