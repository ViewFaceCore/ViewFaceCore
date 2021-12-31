#define STDCALL _stdcall
#define View_Api extern "C" __declspec(dllexport)

typedef void(STDCALL* LogCallBack)(const char* logText);
extern LogCallBack logger;