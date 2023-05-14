#pragma once

#define NO_COPY_ALLOWED(c)				\
	c(const c&) = delete;				\
	c& operator =(const c&) = delete;

#define NO_MOVE_ALLOWED(c)				\
	c(const c&&) = delete;				\
	c& operator =(const c&&) = delete;

namespace Tracer
{
	// mark variables as used
	inline void MarkVariablesUsed(...) {}

	// error handling
	void FatalError(const char* fmt, ...);
}
