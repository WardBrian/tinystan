#ifndef FFISTAN_DEFINES_H
#define FFISTAN_DEFINES_H

#if defined _WIN32 || defined __MINGW32__
#define FFISTAN_ON_WINDOWS 1
else
#define FFISTAN_ON_WINDOWS 0
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
