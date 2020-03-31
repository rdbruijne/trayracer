// Project
#include "App/App.h"
#include "GUI/GuiHelpers.h"
#include "OpenGL/Input.h"
#include "OpenGL/Window.h"
#include "Optix/OptixHelpers.h"
#include "Optix/Renderer.h"
#include "Resources/Scene.h"
#include "Utility/Stopwatch.h"
#include "Utility/Utility.h"

// magic_enum
#include "magic_enum/magic_enum.hpp"

// C++
#include <iostream>

int main(int argc, char** argv)
{
	// init OptiX
	const bool initOptix = Tracer::InitOptix();
	if(!initOptix)
	{
		printf("Failed to init OptiX.\n");
		return -1;
	}
	printf("Successfully initialized OptiX.\n");

	const int2 renderResolution = make_int2(1280, 720);

	// create renderer
	Tracer::Renderer* renderer = new Tracer::Renderer();

	// create window
	Tracer::Window* window = new Tracer::Window("Tracer", renderResolution);

	// init GUI
	Tracer::GuiHelpers::Init(window);
	Tracer::GuiHelpers::renderer = renderer;

	// create app
	Tracer::App* app = new Tracer::App();
	app->Init(renderer, window);

	// timer
	Tracer::Stopwatch stopwatch;
	int64_t elapsedNs = 0;

	bool showGui = false;

	// main loop
	while (!window->IsClosed())
	{
		// user input
		window->UpdateInput();

		if(window->WasKeyPressed(Tracer::Input::Keys::Escape))
			break;

		// update the app
		app->Tick(renderer, window, static_cast<float>(elapsedNs) * 1e-6f);

		// build the scene
		if(app->GetScene()->IsDirty())
		{
			renderer->BuildScene(app->GetScene());
			app->GetScene()->ResetDirtyFlag();
		}

		// run OptiX
		renderer->RenderFrame(window->GetRenderTexture());

		// run window shaders
		window->Display();

		// display GUI
		if(window->WasKeyPressed(Tracer::Input::Keys::F4))
			showGui = !showGui;

		if(showGui)
			Tracer::GuiHelpers::Draw();

		// swap buffers
		window->SwapBuffers();

		// update the title bar
		window->SetTitle(Tracer::format("Tracer - %.1f ms - %.1f FPS - %s - %i samples",
										static_cast<double>(elapsedNs) * 1e-3,
										1e6f / elapsedNs,
										std::string(magic_enum::enum_name(renderer->GetRenderMode())).c_str(),
										renderer->SampleCount()));

		// update timer
		elapsedNs = stopwatch.GetElapsedTimeNS();
		stopwatch.Reset();
	}

	// cleanup
	app->DeInit(renderer, window);
	delete app;

	Tracer::GuiHelpers::DeInit();
	delete window;

	return 0;
}
