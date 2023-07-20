// Project
#include "App.h"

// Tracer
#include "Tracer/App/RunApp.h"
#include "Tracer/GUI/GuiHelpers.h"
#include "Tracer/GUI/MainGui.h"
#include "Tracer/Logging/Logger.h"
#include "Tracer/Logging/ConsoleLogStream.h"
#include "Tracer/Logging/DebuggerLogStream.h"
#include "Tracer/Logging/FileLogStream.h"

using namespace Tracer;

int main(int /*argc*/, char** /*argv*/)
{
	const std::string appName = "TrayRacer Bench";

	// loggers
	Logger::Attach(std::make_shared<ConsoleLogStream>(), Logger::Severity::All);
	Logger::Attach(std::make_shared<DebuggerLogStream>(), Logger::Severity::All);
	Logger::Attach(std::make_shared<FileLogStream>(appName + ".log"), Logger::Severity::All);

	// register GUI windows
	GuiHelpers::Register<MainGui>(Input::Keys::F1);

	// run the app
	Benchmark::App app;
	return RunApp(&app, appName, make_int2(1920, 1080));
}
