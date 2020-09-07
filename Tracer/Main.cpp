// Project
#include "App/App.h"
#include "GUI/GuiHelpers.h"
#include "OpenGL/Input.h"
#include "OpenGL/Window.h"
#include "Renderer/Renderer.h"
#include "Renderer/Scene.h"
#include "Utility/Logger.h"
#include "Utility/LogStream.h"
#include "Utility/Stopwatch.h"
#include "Utility/Utility.h"

// GUI windows
#include "GUI/GuiHelpers.h"
#include "GUI/MainGui.h"

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
			SetConsoleTextAttribute(consoleHandle, info.wAttributes & 0xF);
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
		void Write(Tracer::Logger::Severity severity, const std::string& message) override
		{
			OutputDebugStringA(message.c_str());
			OutputDebugStringA("\n");
		}
	};
}



int main(int argc, char** argv)
{
	try
	{
		// loggers
		Tracer::Logger::Attach(std::make_shared<ConsoleLogSteam>(), Tracer::Logger::Severity::All);
		Tracer::Logger::Attach(std::make_shared<DebuggerLogStream>(), Tracer::Logger::Severity::All);

		// create renderer
		Tracer::Renderer* renderer = new Tracer::Renderer();

		// create window
		const int2 renderResolution = make_int2(1920, 1080);
		Tracer::Window* window = new Tracer::Window("TrayRacer", renderResolution);

		// init GUI
		Tracer::GuiHelpers::Init(window);
		Tracer::MainGui::Get()->SetEnabled(true);

		// create app
		Tracer::App* app = new Tracer::App();
		app->Init(renderer, window);

		// timer
		Tracer::Stopwatch stopwatch;
		float frameTimeMs = 0;

		// init GUI data
		Tracer::GuiHelpers::camNode  = app->GetCameraNode();
		Tracer::GuiHelpers::renderer = renderer;
		Tracer::GuiHelpers::scene    = app->GetScene();
		Tracer::GuiHelpers::window   = window;

		// main loop
		while(!window->IsClosed())
		{
			// begin new frame
			Tracer::GuiHelpers::BeginFrame();

			// user input
			window->UpdateInput();

			if(window->WasKeyPressed(Tracer::Input::Keys::Escape))
				break;

			// update the app
			app->Tick(renderer, window, frameTimeMs * 1e-3f);

			// build the scene
			Tracer::Stopwatch buildTimer;
			if(app->GetScene()->IsDirty())
			{
				renderer->BuildScene(app->GetScene());
				app->GetScene()->MarkClean();
			}
			const float buildTime = buildTimer.ElapsedMs();

			// run Optix
			renderer->RenderFrame(window->RenderTexture());

			// run window shaders
			window->Display();

			// update GUI
			Tracer::MainGui::Get()->UpdateStats(frameTimeMs, buildTime);

			// display GUI
			if(window->WasKeyPressed(Tracer::Input::Keys::F1))
				Tracer::MainGui::Get()->SetEnabled(!Tracer::MainGui::Get()->IsEnabled());
			Tracer::MainGui::Get()->Draw();

			// finalize GUI
			Tracer::GuiHelpers::EndFrame();

			// swap buffers
			window->SwapBuffers();

			// update timer
			frameTimeMs = stopwatch.ElapsedMs();
			stopwatch.Reset();
		}

		// cleanup
		app->DeInit(renderer, window);
		delete app;

		Tracer::GuiHelpers::DeInit();
		delete renderer;
		delete window;
	}
	catch(const std::exception& e)
	{
		Tracer::Logger::Error(e.what());
		std::cerr << e.what() << std::endl;
		assert(false);
	}

	return 0;
}
