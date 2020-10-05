#pragma once

#include "Renderer/Scene.h"
#include "Resources/CameraNode.h"

// C++
#include <memory>

namespace Tracer
{
	class Renderer;
	class Window;
	class App
	{
	public:
		App() = default;
		virtual ~App() = default;

		virtual void Init(Tracer::Renderer* renderer, Tracer::Window* window) = 0;
		virtual void DeInit(Tracer::Renderer* renderer, Tracer::Window* window) = 0;
		virtual void Tick(Tracer::Renderer* renderer, Tracer::Window* window, float dt) = 0;

		virtual Scene* GetScene() { return mScene.get(); }
		virtual Scene* GetScene() const { return mScene.get(); }

		virtual CameraNode* GetCameraNode() { return &mCamera; }

	protected:
		std::unique_ptr<Scene> mScene = nullptr;
		CameraNode mCamera;
	};
}
