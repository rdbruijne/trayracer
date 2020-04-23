#pragma once

// Project
#include "App/ControlScheme.h"
#include "Resources/CameraNode.h"

// C++
#include <memory>

namespace Tracer
{
	class Renderer;
	class Scene;
	class Window;
	class App
	{
	public:
		void Init(Renderer* renderer, Window* window);
		void DeInit(Renderer* renderer, Window* window);
		void Tick(Renderer* renderer, Window* window, float dt);

		Scene* GetScene() { return mScene.get(); }
		Scene* GetScene() const { return mScene.get(); }

		CameraNode* GetCameraNode() { return &mCamera; }

	private:
		void CreateScene();

		std::unique_ptr<Scene> mScene = nullptr;

		// Camera
		CameraNode mCamera;
		ControlScheme mControlScheme;
	};
}
