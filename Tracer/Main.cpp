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
#include "GUI/CameraWindow.h"
#include "GUI/DebugWindow.h"
#include "GUI/MaterialWindow.h"
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
			{ Tracer::MaterialWindow::Get(), Tracer::Input::Keys::F4 },
			{ Tracer::DebugWindow::Get(),    Tracer::Input::Keys::F10 }
		};

		// default enabled windows
		Tracer::StatWindow::Get()->SetEnabled(true);
		Tracer::RendererWindow::Get()->SetEnabled(true);

		// create app
		Tracer::App* app = new Tracer::App();
		app->Init(renderer, window);

		// timer
		Tracer::Stopwatch stopwatch;
		float frameTimeMs = 0;

		// main loop
		while(!window->IsClosed())
		{
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
			Tracer::StatWindow::Get()->mBuildTimeMs = buildTimer.ElapsedMs();

			// run Optix
			renderer->RenderFrame(window->RenderTexture());

			// run window shaders
			window->Display();

			// update GUI
			Tracer::CameraWindow::Get()->mCamNode = app->GetCameraNode();
			Tracer::RendererWindow::Get()->mRenderer = renderer;
			Tracer::RendererWindow::Get()->mWindow = window;
			Tracer::StatWindow::Get()->mFrameTimeMs = frameTimeMs;
			Tracer::StatWindow::Get()->mRenderer = renderer;

			// toggle GUI
			bool anyGuiWindow = false;
			for(auto& w : guiWindows)
			{
				if(window->WasKeyPressed(w.toggleKey))
					w.window->SetEnabled(!w.window->IsEnabled());
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
