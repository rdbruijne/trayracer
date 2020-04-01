#pragma once

// Project
#include "App/CameraNode.h"
#include "App/ControlScheme.h"
#include "Resources/Material.h"
#include "Resources/Mesh.h"

// C++
#include <memory>
#include <vector>

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
		void Tick(Renderer* renderer, Window* window, float dt = 1.f / 60.f);

		Scene* GetScene() { return mScene.get(); }
		Scene* GetScene() const { return mScene.get(); }

	private:
		void CreateScene();

		std::unique_ptr<Scene> mScene = nullptr;

		// Camera
		CameraNode mCamera;
		ControlScheme mControlScheme;

		// objects
		std::vector<std::shared_ptr<Material>> mMaterials;
		std::vector<std::shared_ptr<Mesh>> mMeshes;
	};
}
