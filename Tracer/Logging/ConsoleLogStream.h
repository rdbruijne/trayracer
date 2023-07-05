#pragma once

// Project
#include "Logging/Logger.h"

namespace Tracer
{
	class ConsoleLogStream : public Logger::Stream
	{
	public:
		void Write(Logger::Severity severity, const std::string& message) override;
	};
}
