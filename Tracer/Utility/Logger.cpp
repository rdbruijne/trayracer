#include "Logger.h"

// Project
#include "Utility/LogStream.h"
#include "Utility/Utility.h"

// C++
#include <cstdarg>
#include <ctime>
#include <memory>

namespace Tracer
{
	void Logger::Attach(std::shared_ptr<LogStream> stream, Logger::Severity severities)
	{
		auto add = [stream, severities](Severity sev)
		{
			if(msStreams.find(sev) == msStreams.end())
				msStreams[sev] = {};

			auto& s = msStreams[sev];
			if((severities & sev) == sev)
				if(std::find(s.begin(), s.end(), stream) == s.end())
					s.push_back(stream);
		};

		add(Severity::Debug);
		add(Severity::Info);
		add(Severity::Warning);
		add(Severity::Error);
	}



	void Logger::Detach(std::shared_ptr<LogStream> stream, Logger::Severity severities)
	{
		auto remove = [stream, severities](Severity sev)
		{
			auto& s = msStreams[sev];
			if((severities & sev) == sev)
			{
				auto it = std::find(s.begin(), s.end(), stream);
				if(it != s.end())
					s.erase(it);
			}
		};

		remove(Severity::Debug);
		remove(Severity::Info);
		remove(Severity::Warning);
		remove(Severity::Error);
	}



	void Logger::HandleLog(Severity severity, const char* fmt, ...)
	{
		auto& stream = msStreams[severity];
		if(!stream.empty())
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

			// pass message
			for(auto s : stream)
				s->Write(severity, formatted.get());
		}
	}
}
