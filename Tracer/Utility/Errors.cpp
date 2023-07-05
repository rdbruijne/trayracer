#include "Errors.h"

// Project
#include "Logging/Logger.h"

// C++
#include <cassert>

// Windows
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

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
