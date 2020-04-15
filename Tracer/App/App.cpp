#include "App/App.h"

// Project
#include "App/OrbitCameraController.h"
#include "OpenGL/Window.h"
#include "Optix/Renderer.h"
#include "Resources/Scene.h"
#include "Utility/Importer.h"
#include "Utility/LinearMath.h"

namespace Tracer
{
	void App::Init(Renderer* renderer, Window* window)
	{
		CreateScene();
		mCamera = CameraNode(make_float3(-5, 5, -6), make_float3(0, 0, 0), make_float3(0, 1, 0), 90.f * DegToRad);
		renderer->SetCamera(mCamera.Position, normalize(mCamera.Target - mCamera.Position), mCamera.Up, mCamera.Fov);
	}



	void App::DeInit(Renderer* renderer, Window* window)
	{
	}



	void App::Tick(Renderer* renderer, Window* window, float dt)
	{
		// handle camera controller
		if(OrbitCameraController::HandleInput(mCamera, &mControlScheme, window))
			renderer->SetCamera(mCamera.Position, normalize(mCamera.Target - mCamera.Position), mCamera.Up, mCamera.Fov);

		// camera target picker
		if(window->IsKeyDown(Input::Keys::T) && window->WasKeyPressed(Input::Keys::Mouse_Left))
		{
			const float2 cursorPos = window->CursorPos();
			const uint2 cursorPosU2 = make_uint2(static_cast<uint32_t>(cursorPos.x), static_cast<uint32_t>(cursorPos.y));
			const RayPickResult result = renderer->PickRay(cursorPosU2);

			mCamera.Target = mCamera.Position + result.rayDir * result.dst;
			renderer->SetCamera(mCamera.Position, normalize(mCamera.Target - mCamera.Position), mCamera.Up, mCamera.Fov);
		}
	}



	void App::CreateScene()
	{
		mScene = std::make_unique<Scene>();
		mScene->AddModel(ImportModel("models/toad/toad.obj"));
		//mScene->AddModel(ImportModel("D:/3d_models/Scenes/sponza_crytek/Sponza_Crytek.obj"));
	}
}
