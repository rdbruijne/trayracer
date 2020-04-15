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

// GUI windows
#include "GUI/CameraWindow.h"
#include "GUI/DebugWindow.h"
#include "GUI/RendererWindow.h"
#include "GUI/StatWindow.h"

// C++
#include <iostream>

namespace
{
	struct WindowRegistration
	{
		Tracer::GuiWindow* window;
		Tracer::Input::Keys toggleKey;
	};
}



int main(int argc, char** argv)
{
	try
	{
		// init OptiX
		const bool initOptix = Tracer::InitOptix();
		if(!initOptix)
		{
			printf("Failed to init OptiX.\n");
			return -1;
		}
		printf("Successfully initialized OptiX.\n");

		const int2 renderResolution = make_int2(1920, 1080);

		// create renderer
		Tracer::Renderer* renderer = new Tracer::Renderer();

		// create window
		Tracer::Window* window = new Tracer::Window("TrayRacer", renderResolution);

		// init GUI
		Tracer::GuiHelpers::Init(window);

		std::vector<WindowRegistration> guiWindows =
		{
			{ Tracer::StatWindow::Get(),     Tracer::Input::Keys::F1 },
			{ Tracer::RendererWindow::Get(), Tracer::Input::Keys::F2 },
			{ Tracer::CameraWindow::Get(),   Tracer::Input::Keys::F3 },
			{ Tracer::DebugWindow::Get(),    Tracer::Input::Keys::F10 }
		};

		// create app
		Tracer::App* app = new Tracer::App();
		app->Init(renderer, window);

		// timer
		Tracer::Stopwatch stopwatch;
		int64_t elapsedNs = 0;

		// main loop
		while(!window->IsClosed())
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
			renderer->RenderFrame(window->RenderTexture());

			// run window shaders
			window->Display();

			// update GUI
			Tracer::CameraWindow::Get()->mCamNode = app->GetCameraNode();
			Tracer::RendererWindow::Get()->mRenderer = renderer;
			Tracer::StatWindow::Get()->mFrameTimeNs = elapsedNs;
			Tracer::StatWindow::Get()->mRenderer = renderer;

			// toggle GUI
			bool anyGuiWindow = false;
			for(auto& w : guiWindows)
			{
				if(window->WasKeyPressed(w.toggleKey))
					w.window->Enable(!w.window->IsEnabled());
				anyGuiWindow = anyGuiWindow || w.window->IsEnabled();
			}

			// display GUI
			if(anyGuiWindow)
			{
				Tracer::GuiHelpers::BeginFrame();
				for(auto& w : guiWindows)
					w.window->Draw();
				Tracer::GuiHelpers::EndFrame();
			}

			// swap buffers
			window->SwapBuffers();

			// update timer
			elapsedNs = stopwatch.ElapsedNS();
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
