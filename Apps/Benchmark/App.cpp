#include "App.h"

// Project
#include "FlyCameraController.h"
#include "RecorderGui.h"

// Tracer
#include "Tracer/FileIO/SceneFile.h"
#include "Tracer/GUI/GuiHelpers.h"
#include "Tracer/GUI/MainGui.h"
#include "Tracer/OpenGL/Shader.h"
#include "Tracer/OpenGL/Window.h"
#include "Tracer/Renderer/Scene.h"
#include "Tracer/Renderer/Renderer.h"
#include "Tracer/Optix/Denoiser.h"

using namespace Tracer;

namespace Benchmark
{
	void App::Init(Renderer* renderer, Window* window)
	{
		// set camera
		mCamera = CameraNode(make_float3(0, 0, -1), make_float3(0, 0, 0), make_float3(0, 1, 0), 90.f * DegToRad);
		renderer->SetCamera(mCamera);

		// load scene
		mScene = std::make_unique<Scene>();
		SceneFile::Load("../Resources/scenes/bistro_exterior.json", mScene.get(), mScene->GetSky().get(), &mCamera, renderer, window);

		// render settings
		KernelSettings settings = renderer->Settings();
		settings.multiSample = 8;
		settings.maxDepth = 4;
		renderer->SetStaticNoise(true);
		renderer->SetSettings(settings);
		renderer->GetDenoiser()->SetEnabled(true);
		renderer->GetDenoiser()->SetSampleTreshold(0);

		// start playback
		mPlayTime = 0;

		// set post stack
		std::shared_ptr<Shader> tonemap = std::make_shared<Shader>("Tone Mapping", "glsl/Tonemap.frag", Shader::SourceType::File);
		tonemap->Set("tonemapMethod", 1); // Filmic
		window->SetPostStack({tonemap});

		// load camera path
		mCameraPath.Load(RecorderGui::PathFile);

		// init gui
		GuiHelpers::Register<RecorderGui>(Input::Keys::F2);
		GuiWindow::Get<RecorderGui>()->SetPath(&mCameraPath);
	}



	void App::DeInit(Renderer* /*renderer*/, Window* /*window*/)
	{
	}



	void App::Tick(Renderer* renderer, Window* window, float dt)
	{
		FlyCameraController::HandleInput(mCamera, window, dt);
		renderer->SetCamera(mCamera);

		// update gui
		RecorderGui* recorderGui = GuiWindow::Get<RecorderGui>();
		recorderGui->SetPlayTime(mPlayTime);

		// start/stop playback
		if(mPlayTime < 0)
		{
			if(recorderGui->StartPlayback())
			{
				mPlayTime = 0;
				mBenchFrameCount = 0;
				mBenchTime = 0;
				mBenchStats = {};
			}
		}
		else if(recorderGui->StopPlayback())
		{
			recorderGui->SetBenchmarkResult(mBenchFrameCount, mBenchTime * 1e3f, mBenchStats);
			mPlayTime = -1;
		}

		// playback
		if(mPlayTime >= 0)
		{
			std::optional<Tracer::CameraNode> camNode = mCameraPath.Playback(mPlayTime);
			if(!camNode)
			{
				recorderGui->SetBenchmarkResult(mBenchFrameCount, mBenchTime * 1e3f, mBenchStats);
				mPlayTime = -1;
			}
			else
			{
				if(mPlayTime != 0)
				{
					// record last frame data
					mBenchFrameCount++;
					mBenchTime += dt;
					mBenchStats += renderer->Statistics();
				}

				mCamera = camNode.value();
				mPlayTime += dt;
			}
		}

		// ray picker
		if(window->WasKeyPressed(Input::Keys::Mouse_Left))
		{
			const float2 cursorPos = window->CursorPos();
			const int2 cursorPosI2 = make_int2(static_cast<int32_t>(cursorPos.x), static_cast<int32_t>(cursorPos.y));

			// camera target
			if(window->IsKeyDown(Input::Keys::T))
			{
				const RayPickResult result = renderer->PickRay(cursorPosI2);
				mCamera.SetTarget(mCamera.Position() + result.rayDir * result.tmax);
			}

			// material editor
			if(window->IsKeyDown(Input::Keys::M))
			{
				const RayPickResult result = renderer->PickRay(cursorPosI2);
				GuiWindow::Get<MainGui>()->SelectMaterial(mScene->GetMaterial(result.instIx, result.primIx));
			}
		}

		// autofocus
		const RayPickResult raypick = renderer->PickRay(window->Resolution() / 2);
		mCamera.SetFocalDist(raypick.tmax);
	}
}
