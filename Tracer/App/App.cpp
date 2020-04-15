#include "App/App.h"

// Project
#include "App/OrbitCameraController.h"
#include "Gui/GuiHelpers.h"
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
		if(OrbitCameraController::HandleInput(mCamera, &mControlScheme, window))
			renderer->SetCamera(mCamera.Position, normalize(mCamera.Target - mCamera.Position), mCamera.Up, mCamera.Fov);

		// update GUI
		GuiHelpers::camNode = &mCamera;
	}



	void App::CreateScene()
	{
		mScene = std::make_unique<Scene>();
		mScene->AddModel(ImportModel("models/toad/toad.obj"));
		//mScene->AddModel(ImportModel("D:/3d_models/Scenes/sponza_crytek/Sponza_Crytek.obj"));
	}
}
