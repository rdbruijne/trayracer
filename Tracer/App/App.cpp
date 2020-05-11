#include "App/App.h"

// Project
#include "App/OrbitCameraController.h"
#include "GUI/DebugWindow.h"
#include "OpenGL/Window.h"
#include "Renderer/Scene.h"
#include "Renderer/Renderer.h"
#include "Resources/Instance.h"
#include "Resources/Model.h"
#include "Utility/Importer.h"
#include "Utility/LinearMath.h"
#include "Utility/Utility.h"
#include "GUI/MaterialWindow.h"

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
				MaterialWindow::Get()->selectedMaterial = mScene->GetMaterial(result.instIx, result.primIx);
			}
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
			dw->Set("Dst", format("%f", result.tmax));
			dw->Set("Primitive", format("%d", result.primIx));
			dw->Set("Instance", format("%d", result.instIx));
		}

		// find a toad to rotate
#if false
		for(auto& inst : mScene->Instances())
		{
			if(inst->Name() == "toad")
			{
				inst->SetTransform(inst->Transform() * rotate_y_3x4(static_cast<float>(M_PI) * .05f * dt));
			}
		}
#endif
	}



	void App::CreateScene()
	{
		mScene = std::make_unique<Scene>();

		const int sceneIx = 5;
		switch(sceneIx)
		{
		case 0:
		{
			// toad
			auto toad = ImportModel("models/toad/toad.obj");
			auto toadInst = std::make_shared<Instance>("toad", toad, translate_3x4(0, 1.69f, 0));
			auto toadInst2 = std::make_shared<Instance>("toad2", toad, translate_3x4(-5, 1.69f, 0));

			mScene->AddModel(toad);
			mScene->AddInstance(toadInst);
			mScene->AddInstance(toadInst2);

			// plane
			auto plane = ImportModel("models/plane/plane.obj");
			auto planeInst = std::make_shared<Instance>("plane", plane);

			mScene->AddModel(plane);
			mScene->AddInstance(planeInst);

			mCamera = CameraNode(make_float3(9, 2, 2), make_float3(0, 0, 0), make_float3(0, 1, 0), 90.f * DegToRad);
		}
		break;

		case 1:
		{
			// sponza
			auto sponza = ImportModel("models/sponza/sponza.obj");
			auto sponzaInst = std::make_shared<Instance>("sponza", sponza);

			mScene->AddModel(sponza);
			mScene->AddInstance(sponzaInst);

			mCamera = CameraNode(make_float3(-1350, 150, 0), make_float3(0, 125, 0), make_float3(0, 1, 0), 90.f * DegToRad);
		}
		break;

		case 2:
		{
			// toad
			auto toad = ImportModel("models/toad/toad.obj");
			auto toadInst = std::make_shared<Instance>("toad", toad, translate_3x4(0, 1.69f, 0) * scale_3x4(40) * rotate_y_3x4(3.14f));
			auto toadInst2 = std::make_shared<Instance>("toad2", toad, translate_3x4(10, 1.69f, 10) * scale_3x4(40) * rotate_y_3x4(3.14f));

			mScene->AddModel(toad);
			mScene->AddInstance(toadInst);
			mScene->AddInstance(toadInst2);

			// sponza
			auto sponza = ImportModel("models/sponza/sponza.obj");
			auto sponzaInst = std::make_shared<Instance>("sponza", sponza);

			mScene->AddModel(sponza);
			mScene->AddInstance(sponzaInst);

			//mCamera = CameraNode(make_float3(22, 0, 2), make_float3(0, 0, 0), make_float3(0, 1, 0), 90.f * DegToRad);
			mCamera = CameraNode(make_float3(-1350, 150, 0), make_float3(0, 125, 0), make_float3(0, 1, 0), 90.f * DegToRad);
		}
		break;

		case 3:
		{
			// bistro
			auto bistroIn = ImportModel("models/bistro/BistroInterior.fbx");
			auto bistroEx = ImportModel("models/bistro/BistroExterior.fbx");

			mScene->AddModel(bistroIn);
			mScene->AddModel(bistroEx);

			mScene->AddInstance(std::make_shared<Instance>("bistro interior", bistroIn));
			mScene->AddInstance(std::make_shared<Instance>("bistro exterior", bistroEx));

			mCamera = CameraNode(make_float3(10, 0, 0), make_float3(0, 0, 0), make_float3(0, 1, 0), 90.f * DegToRad);
		}
		break;

		case 4:
		{
			// sun temple
			auto temple = ImportModel("models/sun_temple/SunTemple.fbx");
			mScene->AddModel(temple);
			mScene->AddInstance(std::make_shared<Instance>("sun temple", temple));

			mCamera = CameraNode(make_float3(6, 6, -10), make_float3(-1, 5, 2), make_float3(0, 1, 0), 90.f * DegToRad);
		}
		break;

		case 5:
		{
			// cornell
			auto cornell = ImportModel("models/cornell/cornell.obj");
			mScene->AddModel(cornell);
			mScene->AddInstance(std::make_shared<Instance>("cornell", cornell));

			// toad
			auto toad = ImportModel("models/toad/toad.obj");
			mScene->AddModel(toad);
			mScene->AddInstance(std::make_shared<Instance>("toad", toad, translate_3x4(0, 1.69f, 0) * scale_3x4(50) * rotate_y_3x4(3.14f)));

			mCamera = CameraNode(make_float3(0, 200, -500), make_float3(0, 200, 0), make_float3(0, 1, 0), 90.f * DegToRad);
		}
		break;

		default:
			// no scene
			break;
		}
	}
}
