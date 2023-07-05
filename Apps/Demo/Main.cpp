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
	const std::string appName = "TrayRacer";

	// loggers
	Tracer::Logger::Attach(std::make_shared<ConsoleLogStream>(), Tracer::Logger::Severity::All);
	Tracer::Logger::Attach(std::make_shared<DebuggerLogStream>(), Tracer::Logger::Severity::All);
	Tracer::Logger::Attach(std::make_shared<FileLogStream>(appName + ".log"), Tracer::Logger::Severity::All);

	// register GUI windows
	Tracer::GuiHelpers::Register<Tracer::MainGui>(Tracer::Input::Keys::F1);

	// run the app
	Demo::App demoApp;
	return Tracer::RunApp(&demoApp, appName, make_int2(1920, 1080));
}
