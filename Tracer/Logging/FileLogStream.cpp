#include "Logging/FileLogStream.h"

namespace Tracer
{
	FileLogStream::FileLogStream(const std::string filename)
	{
		mFileStream.open(filename, std::ofstream::out | std::ofstream::trunc);
	}



	void FileLogStream::Write(Logger::Severity /*severity*/, const std::string& message)
	{
		static const std::string newline = "\n";
		mFileStream.write(message.c_str(), static_cast<std::streamsize>(message.length()));
		mFileStream.write(newline.c_str(), static_cast<std::streamsize>(newline.length()));
		mFileStream.flush();
	}
}
