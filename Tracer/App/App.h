#pragma once

// Project
#include "App/CameraNode.h"
#include "App/ControlScheme.h"
#include "CUDA/CudaBuffer.h"
#include "Optix/Optix7.h"
#include "Resources/Mesh.h"

// C++
#include <memory>
#include <vector>

namespace Tracer
{
	class Renderer;
	class Window;
	class App
	{
	public:
		void Init(Renderer* renderer, Window* window);
		void DeInit(Renderer* renderer, Window* window);
		void Tick(Renderer* renderer, Window* window, float dt = 1.f / 60.f);

	private:
		void CreateScene();
		void BuildScene(Renderer* renderer);

		// Scene root object
		OptixTraversableHandle mSceneRoot = 0;

		// Camera
		CameraNode mCamera;
		ControlScheme mControlScheme;

		// temp
		std::shared_ptr<Mesh> mMesh = nullptr;
		CudaBuffer mVertexBuffer;
		CudaBuffer mIndexBuffer;
		CudaBuffer mAccelBuffer;
	};
}
