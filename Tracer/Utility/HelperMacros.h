#pragma once

#ifdef _DEBUG
#define IF_DEBUG(...)			__VA_ARGS__
#define IF_RELEASE(...)
#define BUILD_CONFIGURATION		"DEBUG"
#else
#define IF_DEBUG(...)
#define IF_RELEASE(...)			__VA_ARGS__
#define BUILD_CONFIGURATION		"RELEASE"
#endif

// disable copying
#define NO_COPY_ALLOWED(c)				\
	c(const c&) = delete;				\
	c& operator =(const c&) = delete;
