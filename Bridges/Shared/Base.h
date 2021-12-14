
#define STDCALL
#define View_Api

#ifdef WINDOWS
#define STDCALL _stdcall
#define View_Api extern "C" __declspec(dllexport)
#endif // WIN

#ifdef LINUX
#define STDCALL __attribute__((stdcall))
#define View_Api extern "C"
#endif // LINUX

