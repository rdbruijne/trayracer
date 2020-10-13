#include "RunApp.h"

// Project
#include "App/App.h"
#include "GUI/GuiHelpers.h"
#include "GUI/MainGui.h"
#include "OpenGL/Input.h"
#include "OpenGL/Window.h"
#include "Renderer/Renderer.h"
#include "Renderer/Scene.h"
#include "Utility/Logger.h"
#include "Utility/Stopwatch.h"
#include "Utility/Utility.h"

// C++
#include <iostream>
#include <map>

// Windows
#include <Windows.h>

namespace Tracer
{
	int RunApp(App* app, const std::string& name, const int2& resolution, bool fullscreen, bool useMainGui)
	{
		try
		{
			// create renderer
			Renderer* renderer = new Renderer();

			// create window
			Window* window = new Window(name, resolution, fullscreen);

			// init GUI
			GuiHelpers::Init(window);

			// init app
			app->Init(renderer, window);

			// timer
			Stopwatch stopwatch;
			float frameTimeMs = 0;

			// init GUI data
			GuiHelpers::camNode  = app->GetCameraNode();
			GuiHelpers::renderer = renderer;
			GuiHelpers::scene    = app->GetScene();
			GuiHelpers::window   = window;

			// main loop
			while(!window->IsClosed())
			{
				// begin new frame
				GuiHelpers::BeginFrame();

				// user input
				window->UpdateInput();

				if(window->WasKeyPressed(Input::Keys::Escape))
					break;

				// update the app
				app->Tick(renderer, window, frameTimeMs * 1e-3f);

				// build the scene
				Stopwatch buildTimer;
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
				GuiHelpers::camNode  = app->GetCameraNode();
				GuiHelpers::scene    = app->GetScene();
				if(useMainGui)
				{
					MainGui::Get()->UpdateStats(frameTimeMs, buildTime);
					if(window->WasKeyPressed(Input::Keys::F1))
						MainGui::Get()->SetEnabled(!MainGui::Get()->IsEnabled());
					MainGui::Get()->Draw();
				}

				// finalize GUI
				GuiHelpers::EndFrame();

				// swap buffers
				window->SwapBuffers();

				// update timer
				frameTimeMs = stopwatch.ElapsedMs();
				stopwatch.Reset();
			}
		}
		catch(const std::exception& e)
		{
			Logger::Error(e.what());
			std::cerr << e.what() << std::endl;
			MessageBoxA(NULL, e.what(), "Error", MB_OK | MB_ICONERROR);
			return -1;
		}

		return 0;
	}
}