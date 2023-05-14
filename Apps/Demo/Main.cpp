// Project
#include "App.h"

// Tracer
#include "Tracer/App/RunApp.h"
#include "Tracer/GUI/GuiHelpers.h"
#include "Tracer/GUI/MainGui.h"
#include "Tracer/Utility/Logger.h"

// C++
#include <iostream>
#include <map>

// Windows
#pragma warning(push)
#pragma warning(disable: 4668) // 'symbol' is not defined as a preprocessor macro, replacing with '0' for 'directives'
#pragma warning(disable: 5039) // '_function_': pointer or reference to potentially throwing function passed to `extern C` function under `-EHc`. Undefined behavior may occur if this function throws an exception.
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#pragma warning(pop)

namespace
{
	// Console log stream
	class ConsoleLogStream : public Tracer::Logger::Stream
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



	// Debugger log stream
	class DebuggerLogStream : public Tracer::Logger::Stream
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
	Tracer::Logger::Attach(std::make_shared<ConsoleLogStream>(), Tracer::Logger::Severity::All);
	Tracer::Logger::Attach(std::make_shared<DebuggerLogStream>(), Tracer::Logger::Severity::All);

	// register GUI windows
	Tracer::GuiHelpers::Register<Tracer::MainGui>(Tracer::Input::Keys::F1);

	// run the app
	Demo::App demoApp;
	return Tracer::RunApp(&demoApp, "TrayRacer", make_int2(1920, 1080));
}
