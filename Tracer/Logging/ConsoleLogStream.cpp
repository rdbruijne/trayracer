#include "Logging/ConsoleLogStream.h"

// C++
#include <iostream>

// Windows
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

namespace Tracer
{
	namespace
	{
#pragma warning(push)
#pragma warning(disable: 5264) // 'const' variable is not used
		constexpr WORD Black   = 0x0;
		constexpr WORD Blue    = 0x1;
		constexpr WORD Green   = 0x2;
		constexpr WORD Red     = 0x4;
		constexpr WORD Intense = 0x8;
#pragma warning(pop)

		static constexpr WORD Color(Logger::Severity severity)
		{
			switch(severity)
			{
			case Logger::Severity::Debug:
				return Blue | Green;

			//case Logger::Severity::Info:
			//	return Blue | Green | Red;

			case Logger::Severity::Warning:
				return Green | Red;

			case Logger::Severity::Error:
				return Red;

			default:
				return 0xFFFF;
			}
		}
	}



	void ConsoleLogStream::Write(Logger::Severity severity, const std::string& message)
	{
		static HANDLE consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);

		CONSOLE_SCREEN_BUFFER_INFO info;
		GetConsoleScreenBufferInfo(consoleHandle, &info);
		const WORD color = Color(severity);
		if(color != 0xFFFF)
			SetConsoleTextAttribute(consoleHandle, color);

		std::cout << message;
		if(message.size() == 0 || message.back() != '\n')
			std::cout << '\n';

		if(color != -1)
			SetConsoleTextAttribute(consoleHandle, info.wAttributes & 0xF);
	}
}
