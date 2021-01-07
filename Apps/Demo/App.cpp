#include "App.h"

// Project
#include "OrbitCameraController.h"

// Tracer
#include "Tracer/FileIO/SceneFile.h"
#include "Tracer/GUI/MainGui.h"
#include "Tracer/OpenGL/Window.h"
#include "Tracer/Renderer/Scene.h"
#include "Tracer/Renderer/Renderer.h"
#include "Tracer/Resources/Instance.h"
#include "Tracer/Utility/LinearMath.h"

using namespace Tracer;
namespace Demo
{
	void App::Init(Renderer* renderer, Window* window)
	{
		mScene = std::make_unique<Scene>();

		// default camera
		mCamera = CameraNode(make_float3(0, 0, -1), make_float3(0, 0, 0), make_float3(0, 1, 0), 90.f * DegToRad);

		// update camera
		renderer->SetCamera(mCamera);

		// load a scene
		//SceneFile::Load("../Resources/scenes/toad_on_a_plane.json", mScene.get(), &mCamera, renderer, window);
		MarkVariablesUsed(window);
	}



	void App::DeInit(Renderer* /*renderer*/, Window* /*window*/)
	{
	}



	void App::Tick(Renderer* renderer, Window* window, float /*dt*/)
	{
		// handle camera controller
		OrbitCameraController::HandleInput(mCamera, &mControlScheme, window);

		// ray picker
		if(window->WasKeyPressed(Input::Keys::Mouse_Left))
		{
			const float2 cursorPos = window->CursorPos();
			const int2 cursorPosI2 = make_int2(static_cast<int32_t>(cursorPos.x), static_cast<int32_t>(cursorPos.y));

			// camera focal distance
			if(window->IsKeyDown(Input::Keys::F))
			{
				const RayPickResult result = renderer->PickRay(cursorPosI2);
				mCamera.SetFocalDist(result.tmax);
			}

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

		// set the camera
		renderer->SetCamera(mCamera);
	}
}
