#pragma once

// Project
#include "Utility/Enum.h"

// C++
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace Tracer
{
	class LogStream;
	class Logger
	{
	public:
		// Severity
		enum class Severity
		{
			Debug		= 0x1,
			Info		= 0x2,
			Warning		= 0x4,
			Error		= 0x8,

			// shorthands
			All			= Debug | Info | Warning | Error
		};

		// register logger
		static void Attach(std::shared_ptr<LogStream> stream, Severity severities);
		static void Detach(std::shared_ptr<LogStream> stream, Severity severities);

		// log messages
		template <typename... Args>
		static void Debug(const char* fmt, Args&&... args)
		{
			HandleLog(Severity::Debug, fmt, std::forward<Args>(args)...);
		}

		template <typename... Args>
		static void Info(const char* fmt, Args&&... args)
		{
			HandleLog(Severity::Info, fmt, std::forward<Args>(args)...);
		}

		template <typename... Args>
		static void Warning(const char* fmt, Args&&... args)
		{
			HandleLog(Severity::Warning, fmt, std::forward<Args>(args)...);
		}

		template <typename... Args>
		static void Error(const char* fmt, Args&&... args)
		{
			HandleLog(Severity::Error, fmt, std::forward<Args>(args)...);
		}

	private:
		static void HandleLog(Severity severity, const char* fmt, ...);

		static inline std::map<Severity, std::vector<std::shared_ptr<LogStream>>> msStreams =
		{
			{ Severity::Debug, {} },
			{ Severity::Info, {} },
			{ Severity::Warning, {} },
			{ Severity::Error, {} },
		};
	};

	ENUM_BITWISE_OPERATORS(Logger::Severity);
}
