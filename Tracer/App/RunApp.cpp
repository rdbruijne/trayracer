#include "RunApp.h"

// Project
#include "App/App.h"
#include "FileIO/TextureFile.h"
#include "GUI/GuiHelpers.h"
#include "GUI/MainGui.h"
#include "OpenGL/GLHelpers.h"
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
	int RunApp(App* app, const std::string& name, const int2& resolution, bool fullscreen)
	{
		// initialize OpenGL
		InitGL();

		// create renderer
		Renderer* renderer = new Renderer();

		// create window
		Window* window = new Window();
		window->Open(name, resolution, fullscreen);

		// init GUI
		GuiHelpers::Init(window);

		// init app
		app->Init(renderer, window);

		// timer
		Stopwatch stopwatch;
		float frameTimeMs = 0;

		// init GUI data
		GuiHelpers::Set(app->GetCameraNode());
		GuiHelpers::Set(renderer);
		GuiHelpers::Set(app->GetScene());

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

			// save render?
			std::string savePath;
			if(renderer->SaveRequested(savePath))
			{
				TextureFile::Export(savePath, window->DownloadFramebuffer());
				renderer->ResetSaveRequest();
			}

			// update GUI
			GuiHelpers::Set(app->GetCameraNode());
			GuiHelpers::Set(app->GetScene());
			GuiHelpers::SetFrameTimeMs(frameTimeMs);
			GuiHelpers::DrawGui();

			// finalize GUI
			GuiHelpers::EndFrame();

			// swap buffers
			window->SwapBuffers();

			// update timer
			frameTimeMs = stopwatch.ElapsedMs();
			stopwatch.Reset();
		}

		// clean state
		delete renderer;
		delete window;
		TerminateGL();

		return 0;
	}
}
