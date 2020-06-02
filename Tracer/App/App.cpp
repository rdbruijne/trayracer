#include "App/App.h"

// Project
#include "App/OrbitCameraController.h"
#include "GUI/DebugGui.h"
#include "GUI/MaterialGui.h"
#include "OpenGL/Window.h"
#include "Renderer/Scene.h"
#include "Renderer/Renderer.h"
#include "Resources/Instance.h"

namespace Tracer
{
	void App::Init(Renderer* renderer, Window* window)
	{
		mScene = std::make_unique<Scene>();

		// default camera
		mCamera = CameraNode(make_float3(0, 0, -1), make_float3(0, 0, 0), make_float3(0, 1, 0), 90.f * DegToRad);

		// update camera
		renderer->SetCamera(mCamera);
	}



	void App::DeInit(Renderer* renderer, Window* window)
	{
	}



	void App::Tick(Renderer* renderer, Window* window, float dt)
	{
		// handle camera controller
		OrbitCameraController::HandleInput(mCamera, &mControlScheme, window);

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
				MaterialGui::Get()->mSelectedMaterial = mScene->GetMaterial(result.instIx, result.primIx);
			}
		}

		// set the camera
		renderer->SetCamera(mCamera);

		// find a toad to rotate
		/*
		for(auto& inst : mScene->Instances())
		{
			if(inst->Name() == "toad")
			{
				inst->SetTransform(inst->Transform() * rotate_y_3x4(static_cast<float>(M_PI) * .05f * dt));
			}
		}
		/**/
	}
}
