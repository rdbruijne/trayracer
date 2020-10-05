// Project
#include "App.h"

// Tracer
#include "Tracer/App/RunApp.h"
#include "Tracer/Utility/Logger.h"
#include "Tracer/Utility/LogStream.h"

// C++
#include <iostream>
#include <map>

// Windows
#include <Windows.h>

namespace
{
	class ConsoleLogSteam : public Tracer::LogStream
	{
	public:
		void Write(Tracer::Logger::Severity severity, const std::string& message) override
		{
			static HANDLE consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
			CONSOLE_SCREEN_BUFFER_INFO info;
			GetConsoleScreenBufferInfo(consoleHandle, &info);
			if(msColors.count(severity) != 0)
				SetConsoleTextAttribute(consoleHandle, static_cast<WORD>(msColors[severity]));
			std::cout << message << "\n";
			SetConsoleTextAttribute(consoleHandle, static_cast<WORD>(info.wAttributes & 0xF));
		}

	private:
		inline static int Black   = 0x0;
		inline static int Blue    = 0x1;
		inline static int Green   = 0x2;
		inline static int Red     = 0x4;
		inline static int Intense = 0x8;

		inline static std::map<Tracer::Logger::Severity, int> msColors =
		{
			{ Tracer::Logger::Severity::Debug,   Blue | Green },
			//{ Tracer::Logger::Severity::Info,    Blue | Green | Red},
			{ Tracer::Logger::Severity::Warning, Green | Red },
			{ Tracer::Logger::Severity::Error,   Red }
		};
	};



	class DebuggerLogStream : public Tracer::LogStream
	{
	public:
		void Write(Tracer::Logger::Severity /*severity*/, const std::string& message) override
		{
			OutputDebugStringA(message.c_str());
			OutputDebugStringA("\n");
		}
	};
}



int main(int /*argc*/, char** /*argv*/)
{
	// loggers
	Tracer::Logger::Attach(std::make_shared<ConsoleLogSteam>(), Tracer::Logger::Severity::All);
	Tracer::Logger::Attach(std::make_shared<DebuggerLogStream>(), Tracer::Logger::Severity::All);

	// run the app
	Demo::App demoApp;
	return Tracer::RunApp(&demoApp, "TrayRacer", make_int2(1920, 1080));
}
