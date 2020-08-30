// Project
#include "App/App.h"
#include "GUI/GuiHelpers.h"
#include "OpenGL/Input.h"
#include "OpenGL/Window.h"
#include "Renderer/Renderer.h"
#include "Renderer/Scene.h"
#include "Utility/Stopwatch.h"
#include "Utility/Utility.h"

// GUI windows
#include "GUI/GuiHelpers.h"
#include "GUI/MainGui.h"

// C++
#include <iostream>

int main(int argc, char** argv)
{
	try
	{
		const int2 renderResolution = make_int2(1920, 1080);

		// create renderer
		Tracer::Renderer* renderer = new Tracer::Renderer();

		// create window
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
		std::cerr << e.what() << std::endl;
		assert(false);
	}

	return 0;
}
