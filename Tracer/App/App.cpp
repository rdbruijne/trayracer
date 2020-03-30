#include "App/App.h"

// Project
#include "App/OrbitCameraController.h"
#include "Gui/GuiHelpers.h"
#include "Optix/Renderer.h"
#include "Resources/Scene.h"
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



	void App::Tick(Renderer* renderer, Window* window, float dt /*= 1.f / 60.f*/)
	{
		if(OrbitCameraController::HandleInput(mCamera, &mControlScheme, window))
			renderer->SetCamera(mCamera.Position, normalize(mCamera.Target - mCamera.Position), mCamera.Up, mCamera.Fov);

		// update GUI
		GuiHelpers::camNode = &mCamera;
	}



	void App::CreateScene()
	{
		mScene = std::make_unique<Scene>();

		// material
		mMaterial = std::make_shared<Material>();
		mMaterial->Diffuse = make_float3(.25f, .75f, .25f);
		mScene->AddMaterial(mMaterial);

		// cube
		mMesh = std::make_shared<Mesh>(
			std::vector<float3>
			{
				make_float3(-0.5f, -0.5f, -0.5f),
				make_float3(-0.5f, -0.5f,  0.5f),
				make_float3(-0.5f,  0.5f, -0.5f),
				make_float3(-0.5f,  0.5f,  0.5f),
				make_float3( 0.5f, -0.5f, -0.5f),
				make_float3( 0.5f, -0.5f,  0.5f),
				make_float3( 0.5f,  0.5f, -0.5f),
				make_float3( 0.5f,  0.5f,  0.5f)
			},
			std::vector<float3>(), // ignore normals
			std::vector<float3>(), // ignore texcoords
			std::vector<uint3>
			{
				make_uint3(0, 3, 2),
				make_uint3(0, 1, 3),
				make_uint3(4, 7, 5),
				make_uint3(4, 6, 7),
				make_uint3(0, 5, 1),
				make_uint3(0, 4, 5),
				make_uint3(2, 7, 6),
				make_uint3(2, 3, 7),
				make_uint3(0, 6, 4),
				make_uint3(0, 2, 6),
				make_uint3(1, 7, 3),
				make_uint3(1, 5, 7)
			});
		mScene->AddMesh(mMesh);
	}
}
