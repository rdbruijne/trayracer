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
#include "GUI/CameraGui.h"
#include "GUI/DebugGui.h"
#include "GUI/MaterialGui.h"
#include "GUI/RendererGui.h"
#include "GUI/StatGui.h"

// C++
#include <iostream>

namespace
{
	struct WindowRegistration
	{
		Tracer::BaseGui* window;
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
			{ Tracer::StatGui::Get(),     Tracer::Input::Keys::F1 },
			{ Tracer::RendererGui::Get(), Tracer::Input::Keys::F2 },
			{ Tracer::CameraGui::Get(),   Tracer::Input::Keys::F3 },
			{ Tracer::MaterialGui::Get(), Tracer::Input::Keys::F4 },
			{ Tracer::DebugGui::Get(),    Tracer::Input::Keys::F10 }
		};

		// default enabled windows
		//Tracer::StatGui::Get()->SetEnabled(true);
		Tracer::RendererGui::Get()->SetEnabled(true);
		//Tracer::CameraGui::Get()->SetEnabled(true);
		//Tracer::MaterialGui::Get()->SetEnabled(true);
		//Tracer::DebugGui::Get()->SetEnabled(true);

		// create app
		Tracer::App* app = new Tracer::App();
		app->Init(renderer, window);

		// timer
		Tracer::Stopwatch stopwatch;
		float frameTimeMs = 0;

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
			Tracer::StatGui::Get()->mBuildTimeMs = buildTimer.ElapsedMs();

			// run Optix
			renderer->RenderFrame(window->RenderTexture());

			// run window shaders
			window->Display();

			// update GUI
			Tracer::CameraGui::Get()->mCamNode = app->GetCameraNode();
			Tracer::MaterialGui::Get()->mScene = app->GetScene();
			Tracer::RendererGui::Get()->mCamNode = app->GetCameraNode();
			Tracer::RendererGui::Get()->mRenderer = renderer;
			Tracer::RendererGui::Get()->mScene = app->GetScene();
			Tracer::RendererGui::Get()->mWindow = window;
			Tracer::StatGui::Get()->mFrameTimeMs = frameTimeMs;
			Tracer::StatGui::Get()->mRenderer = renderer;
			Tracer::StatGui::Get()->mScene = app->GetScene();

			// display GUI
			for(auto& w : guiWindows)
			{
				if(window->WasKeyPressed(w.toggleKey))
					w.window->SetEnabled(!w.window->IsEnabled());
				w.window->Draw();
			}

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
