#include "App.h"

// Tracer
#include "Tracer/Controllers/OrbitCameraController.h"
#include "Tracer/FileIO/ModelFile.h"
#include "Tracer/FileIO/SceneFile.h"
#include "Tracer/GUI/MainGui.h"
#include "Tracer/OpenGL/Shader.h"
#include "Tracer/OpenGL/Window.h"
#include "Tracer/Renderer/Scene.h"
#include "Tracer/Renderer/Renderer.h"
#include "Tracer/Resources/Instance.h"
#include "Tracer/Resources/Model.h"
#include "Tracer/Utility/LinearMath.h"

// C++
#include <memory>

using namespace Tracer;
namespace Demo
{
	void App::Init([[maybe_unused]] Renderer* renderer, [[maybe_unused]] Window* window)
	{
		mScene = std::make_unique<Scene>();

		// default camera
		mCamera = CameraNode(make_float3(0, 0, -1), make_float3(0, 0, 0), make_float3(0, 1, 0), 90.f * DegToRad);

		// update camera
		renderer->SetCamera(mCamera);

		// load a scene
		//SceneFile::Load("../Resources/scenes/toad_on_a_plane.json", mScene.get(), mScene->GetSky().get(), &mCamera, renderer, window);

		// add a model
		//std::shared_ptr<Model> toad = ModelFile::Import(mScene.get(), "../Resources/models/toad/toad.obj", "toad");
		//std::shared_ptr<Instance> toadInst = std::make_shared<Instance>("toad", toad, make_float3x4());
		//mScene->Add(toadInst);

		// set post stack
		std::shared_ptr<Shader> tonemap = std::make_shared<Shader>("Tone Mapping", "glsl/Tonemap.frag", Shader::SourceType::File);
		window->SetPostStack({tonemap});
	}



	void App::DeInit([[maybe_unused]] Renderer* renderer, [[maybe_unused]] Window* window)
	{
	}



	void App::Tick([[maybe_unused]] Renderer* renderer, [[maybe_unused]] Window* window, [[maybe_unused]] float dt)
	{
		// handle camera controller
		OrbitCameraController::HandleInput(mCamera, window);

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
