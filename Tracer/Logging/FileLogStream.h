#pragma once

// Project
#include "Logging/Logger.h"

// C++
#include <fstream>

namespace Tracer
{
	class FileLogStream : public Logger::Stream
	{
	public:
		explicit FileLogStream(const std::string filename);

		FileLogStream(const FileLogStream&) = delete;
		FileLogStream(const FileLogStream&&) = delete;
		FileLogStream& operator =(const FileLogStream&) = delete;
		FileLogStream& operator =(const FileLogStream&&) = delete;

		void Write(Logger::Severity severity, const std::string& message) override;

	private:
		std::ofstream mFileStream;
	};
}
