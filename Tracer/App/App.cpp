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

		// load scene
		//mScene->Load("scenes/cornell.json", &mCamera);
		//mScene->Load("scenes/bistro.json", &mCamera);
		//mScene->Load("scenes/sponza.json", &mCamera);
		//mScene->Load("scenes/sponza_toad.json", &mCamera);
		//mScene->Load("scenes/suntemple.json", &mCamera);
		//mScene->Load("scenes/toad_on_a_plane.json", &mCamera);

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
				MaterialGui::Get()->selectedMaterial = mScene->GetMaterial(result.instIx, result.primIx);
			}
		}

		// set the camera
		renderer->SetCamera(mCamera);

		// debug raypick
		DebugGui* debugGui = DebugGui::Get();
		if(debugGui->IsEnabled())
		{
			const float2 cursorPos = window->CursorPos();
			const int2 cursorPosI2 = make_int2(static_cast<int32_t>(cursorPos.x), static_cast<int32_t>(cursorPos.y));
			const RayPickResult result = renderer->PickRay(cursorPosI2);

			debugGui->Set("Pixel", format("%d  %d", cursorPosI2.x, cursorPosI2.y));
			debugGui->Set("Ray origin", format("%.2f  %.2f  %.2f", result.rayOrigin.x, result.rayOrigin.y, result.rayOrigin.z));
			debugGui->Set("Ray dir", format("%.2f  %.2f  %.2f", result.rayDir.x, result.rayDir.y, result.rayDir.z));
			debugGui->Set("Dst", format("%f", result.tmax));
			debugGui->Set("Primitive", format("%d", result.primIx));
			debugGui->Set("Instance", format("%d", result.instIx));
		}

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
