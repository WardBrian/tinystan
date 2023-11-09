#ifndef FFISTAN_DEFINES_H
#define FFISTAN_DEFINES_H

#ifndef FFISTAN_ON_WINDOWS
#define FFISTAN_ON_WINDOWS defined _WIN32 || defined __MINGW32__
#endif

#if FFISTAN_ON_WINDOWS
#if FFISTAN_EXPORT
#define FFISTAN_PUBLIC __declspec(dllexport)
#else
#define FFISTAN_PUBLIC __declspec(dllimport)
#endif
#else
#define FFISTAN_PUBLIC __attribute__((visibility("default")))
#endif

#endif
