#pragma once

// Project
#include "CudaBuffer.h"
#include "Mesh.h"

// OptiX
#include "Optix7.h"

// C++
#include <memory>
#include <vector>

namespace Tracer
{
	class Renderer;
	class Window;

	/*!
	 * @brief App class.
	 *
	 * Controls the scene/control logic.
	 */
	class App
	{
	public:
		/*!
		 * @brief Initialize the app.
		 * @param[in] renderer OptiX renderer.
		 * @param[in] window The render window.
		 */
		void Init(Renderer* renderer, Window* window);

		/*!
		 * @brief Uninitialize the app.
		 * @param[in] renderer OptiX renderer.
		 * @param[in] window The render window.
		 */
		void DeInit(Renderer* renderer, Window* window);

		/*!
		 * @brief Update the app.
		 * @param[in] renderer OptiX renderer.
		 * @param[in] window The render window.
		 * @param[in] dt Frame time (in seconds).
		 */
		void Tick(Renderer* renderer, Window* window, float dt = 1.f / 60.f);

	private:
		/*!
		 * @brief Create the test scene.
		 */
		void CreateScene();

		/*!
		 * @brief Build the test scene.
		 */
		void BuildScene(Renderer* renderer);

		/*! Scene root object. */
		OptixTraversableHandle mSceneRoot = 0;

		//--------------------------------
		// temp
		//--------------------------------
		/*! @{ */
		/*! Test mesh. */
		std::shared_ptr<Mesh> mMesh = nullptr;
		/*! CUDA buffer for the scene's vertex positions. */
		CudaBuffer mVertexBuffer;
		/*! CUDA buffer for the scene's vertex indices. */
		CudaBuffer mIndexBuffer;
		/*! CUDA buffer for the acceleration structure. */
		CudaBuffer mAccelBuffer;
		/*! @} */
	};
}
