namespace Tracer
{
	template <typename... Args>
	static void Logger::Message(Severity severity, const char* fmt, Args&&... args)
	{
		HandleLog(severity, fmt, std::forward<Args>(args)...);
	}



	template <typename... Args>
	static void Logger::Debug(const char* fmt, Args&&... args)
	{
		HandleLog(Severity::Debug, fmt, std::forward<Args>(args)...);
	}



	template <typename... Args>
	static void Logger::Info(const char* fmt, Args&&... args)
	{
		HandleLog(Severity::Info, fmt, std::forward<Args>(args)...);
	}



	template <typename... Args>
	static void Logger::Warning(const char* fmt, Args&&... args)
	{
		HandleLog(Severity::Warning, fmt, std::forward<Args>(args)...);
	}



	template <typename... Args>
	static void Logger::Error(const char* fmt, Args&&... args)
	{
		HandleLog(Severity::Error, fmt, std::forward<Args>(args)...);
	}
}
