#include "Logging/DebuggerLogStream.h"

// Windows
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

namespace Tracer
{
	void DebuggerLogStream::Write(Logger::Severity /*severity*/, const std::string& message)
	{
		OutputDebugStringA(message.c_str());
		OutputDebugStringA("\n");
	}
}
