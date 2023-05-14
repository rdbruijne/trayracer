#include "Errors.h"

// Project
#include "Utility/Logger.h"

// C++
#include <assert.h>

// Windows
#pragma warning(push)
#pragma warning(disable: 4668) // 'symbol' is not defined as a preprocessor macro, replacing with '0' for 'directives'
#pragma warning(disable: 5039) // '_function_': pointer or reference to potentially throwing function passed to `extern C` function under `-EHc`. Undefined behavior may occur if this function throws an exception.
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#pragma warning(pop)

namespace Tracer
{
	void FatalError(const char* fmt, ...)
	{
		// format message
		size_t n = strlen(fmt);
		int final_n = -1;

		std::unique_ptr<char[]> formatted;
		va_list ap;
		do
		{
			n <<= 1;
			formatted.reset(new char[n + 1]);
			va_start(ap, fmt);
			final_n = vsnprintf_s(formatted.get(), n - 1, n - 2, fmt, ap);
			va_end(ap);
		} while(final_n < 0);

		// handle error
		Logger::Error(formatted.get());
		MessageBoxA(NULL, formatted.get(), "Fatal Error", MB_OK | MB_ICONERROR);
		assert(false);
		exit(EXIT_FAILURE);
	}
}
