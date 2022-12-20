#include "RunApp.h"

// Project
#include "App/App.h"
#include "FileIO/SceneFile.h"
#include "FileIO/ModelFile.h"
#include "FileIO/TextureFile.h"
#include "GUI/GuiHelpers.h"
#include "GUI/MainGui.h"
#include "OpenGL/GLHelpers.h"
#include "OpenGL/Input.h"
#include "OpenGL/Shader.h"
#include "OpenGL/Window.h"
#include "Renderer/Renderer.h"
#include "Renderer/Scene.h"
#include "Resources/Instance.h"
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
	namespace
	{
		void HandleDrops(App* app, Renderer* renderer, Window* window)
		{
			const std::vector<std::string> drops = window->Drops();
			window->ClearDrops();

			for(const std::string& d : drops)
			{
				const std::string ext = FileExtension(d);
				if(ToLower(ext) == ".json")
				{
					const int clearScene = MessageBoxA(NULL, "Override", "Override existing scene?", MB_YESNO | MB_ICONQUESTION);
					if(clearScene == IDYES)
					{
						app->GetScene()->Clear();
						SceneFile::Load(d, app->GetScene(), app->GetScene()->GetSky().get(), app->GetCameraNode(), renderer, window);
					}
					else
					{
						SceneFile::Load(d, app->GetScene(), nullptr, nullptr, nullptr, nullptr);
					}
				}
				else if(ModelFile::Supports(d))
				{
					std::shared_ptr<Model> model = ModelFile::Import(app->GetScene(), d);
					std::shared_ptr<Instance> inst = std::make_shared<Instance>(FileName(d), model, make_float3x4());
					app->GetScene()->Add(inst);
				}
			}
		}
	}



	int RunApp(App* app, const std::string& windowTitle, const int2& resolution, bool fullscreen)
	{
		// check params
		if(!app)
			FatalError("No app specified");

		if((resolution.x <= 0) || (resolution.y <= 0))
			FatalError("Invalid resolution: %i x %i", resolution.x, resolution.y);

		// initialize OpenGL
		InitGL();

		// create renderer
		Renderer* renderer = new Renderer();

		// create window
		Window* window = new Window();
		window->Open(windowTitle, resolution, fullscreen);

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
			renderer->UpdateScene(app->GetScene());
			const float buildTime = buildTimer.ElapsedMs();

			// run Optix
			renderer->RenderFrame(window->RenderTexture());

			// run window shaders
			window->Display();

			// handle drops
			if(window->HasDrops())
				HandleDrops(app, renderer, window);

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
