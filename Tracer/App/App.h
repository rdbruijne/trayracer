#pragma once

// Project
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
		struct Camera
		{
			Camera() = default;
			explicit Camera(const float3& position, const float3& target, const float3& up) : Position(position), Target(target), Up(up) {}

			float3 Position = make_float3(0, 0, -1);
			float3 Target = make_float3(0, 0, 0);
			float3 Up = make_float3(0, 1, 0);
		} mCamera;

		// temp
		std::shared_ptr<Mesh> mMesh = nullptr;
		CudaBuffer mVertexBuffer;
		CudaBuffer mIndexBuffer;
		CudaBuffer mAccelBuffer;
	};
}
