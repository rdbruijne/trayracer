#include "App/App.h"

// Project
#include "App/OrbitCameraController.h"
#include "GUI/DebugWindow.h"
#include "OpenGL/Window.h"
#include "Optix/Renderer.h"
#include "Resources/Scene.h"
#include "Utility/Importer.h"
#include "Utility/LinearMath.h"
#include "Utility/Utility.h"

namespace Tracer
{
	void App::Init(Renderer* renderer, Window* window)
	{
		CreateScene();

		renderer->SetCamera(mCamera);
	}



	void App::DeInit(Renderer* renderer, Window* window)
	{
	}



	void App::Tick(Renderer* renderer, Window* window, float dt)
	{
		// handle camera controller
		OrbitCameraController::HandleInput(mCamera, &mControlScheme, window);

		// camera target picker
		if(window->IsKeyDown(Input::Keys::T) && window->WasKeyPressed(Input::Keys::Mouse_Left))
		{
			const float2 cursorPos = window->CursorPos();
			const int2 cursorPosI2 = make_int2(static_cast<int32_t>(cursorPos.x), static_cast<int32_t>(cursorPos.y));
			const RayPickResult result = renderer->PickRay(cursorPosI2);

			mCamera.SetTarget(mCamera.Position() + result.rayDir * result.dst);
		}

		// set the camera
		renderer->SetCamera(mCamera);

		// debug raypick
		DebugWindow* dw = DebugWindow::Get();
		if(dw->IsEnabled())
		{
			const float2 cursorPos = window->CursorPos();
			const int2 cursorPosI2 = make_int2(static_cast<int32_t>(cursorPos.x), static_cast<int32_t>(cursorPos.y));
			const RayPickResult result = renderer->PickRay(cursorPosI2);

			dw->Set("Pixel", format("%d  %d", cursorPosI2.x, cursorPosI2.y));
			dw->Set("Ray origin", format("%.2f  %.2f  %.2f", result.rayOrigin.x, result.rayOrigin.y, result.rayOrigin.z));
			dw->Set("Ray dir", format("%.2f  %.2f  %.2f", result.rayDir.x, result.rayDir.y, result.rayDir.z));
			dw->Set("Dst", format("%f", result.dst));
			dw->Set("Target", format("%d", result.objectID));
		}
	}



	void App::CreateScene()
	{
		mScene = std::make_unique<Scene>();

		// toad
		mScene->AddModel(ImportModel("models/toad/toad.obj"));
		mCamera = CameraNode(make_float3(-5, 5, -6), make_float3(0, 0, 0), make_float3(0, 1, 0), 90.f * DegToRad);

		// sponza
		//mScene->AddModel(ImportModel("models/sponza/sponza.obj"));
		//mCamera = CameraNode(make_float3(-1350, 150, 0), make_float3(0, 125, 0), make_float3(0, 1, 0), 90.f * DegToRad);
	}
}
