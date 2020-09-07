#pragma once

// Project
#include "Utility/Logger.h"

// C++
#include <string>

namespace Tracer
{
	class LogStream
	{
	public:
		virtual ~LogStream() = default;

		virtual void Write(Logger::Severity severity, const std::string& message) = 0;
	};
}
