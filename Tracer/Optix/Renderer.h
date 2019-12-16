#pragma once

// Project
#include "CommonStructs.h"
#include "CudaBuffer.h"

// C++
#include <string>
#include <vector>

/*! Tracer namespace */
namespace Tracer
{
	/*!
	 * @brief OptiX renderer.
	 */
	class Renderer
	{
	public:
		/*!
		 * @brief Construct an OptiX renderer.
		 * @param[in] resolution The render resolution.
		 */
		explicit Renderer(const int2& resolution);

		/*!
		 * @brief Deconstruct a renderer.
		 */
		~Renderer();

		/*!
		 * @brief Render a single frame.
		 */
		void RenderFrame();

		/*!
		 * @brief Download the color buffer
		 * @param[out] dstPixels The downloaded pixel buffer.
		 */
		void DownloadPixels(std::vector<uint32_t>& dstPixels);

		/*!
		 * @brief Resize the renderer.
		 * @param[in] resolution The render resolution.
		 */
		void Resize(const int2& resolution);

		/*!
		 * @brief Set the scene root.
		 * @param[in] sceneRoot Handle to the scene root object.
		 */
		void SetSceneRoot(OptixTraversableHandle sceneRoot);

		/*!
		 * @brief Set the camera info.
		 */
		void SetCamera(float3 cameraPos, float3 cameraTarget, float3 cameraUp);

		/*!
		 * @brief Get the OptiX context
		 */
		OptixDeviceContext GetOptixDeviceContext()
		{
			return mOptixContext;
		}

	private:
		/*!
		 * @brief Create and configures OptiX device context.
		 */
		void CreateContext();

		/*!
		 * @brief Create the OptiX module.
		 */
		void CreateModule();

		/*!
		 * @brief Create the ray gen programs.
		 */
		void CreateRaygenPrograms();

		/*!
		 * @brief Create the miss programs.
		 */
		void CreateMissPrograms();

		/*!
		 * @brief Create the hit group programs.
		 */
		void CreateHitgroupPrograms();

		/*!
		 * @brief Create the render pipeline.
		 */
		void CreatePipeline();

		/*!
		 * @brief Build the shader binding table.
		 */
		void BuildShaderBindingTable();

		/*! @{ Render buffer */
		CudaBuffer mColorBuffer;
		/*! @} */

		/*! @{ Launch parameters */
		LaunchParams mLaunchParams;
		/*! CUDA buffer for the launch parameters */
		CudaBuffer   mLaunchParamsBuffer;
		/*! @} */

		/*! @{ CUDA device context */
		CUcontext      mCudaContext;
		/*! CUDA stream */
		CUstream       mStream;
		/*! CUDA device properties */
		cudaDeviceProp mDeviceProperties;
		/*! @} */

		/*! @{ OptiX module */
		OptixModule               mModule;
		/*! Compile options for @ref mModule */
		OptixModuleCompileOptions mModuleCompileOptions;
		/*! @} */

		/*! @{ OptiX pipeline */
		OptixPipeline               mPipeline;
		/*! Compile options for @ref mPipeline */
		OptixPipelineCompileOptions mPipelineCompileOptions;
		/*! Link options for @ref mPipeline */
		OptixPipelineLinkOptions    mPipelineLinkOptions;
		/*! @} */

		/*! @{ The OptiX device context */
		OptixDeviceContext mOptixContext;
		/*! @} */

		/*! @{ Ray generation programs */
		std::vector<OptixProgramGroup> mRayGenPrograms;
		/*! Cuda buffer for @ref mRayGenPrograms */
		CudaBuffer mRaygenRecordsBuffer;

		/*! Miss programs */
		std::vector<OptixProgramGroup> mMissPrograms;
		/*! Cuda buffer for @ref mMissPrograms */
		CudaBuffer mMissRecordsBuffer;

		/*! Hit programs */
		std::vector<OptixProgramGroup> mHitgroupPrograms;
		/*! Cuda buffer for @ref mHitgroupPrograms */
		CudaBuffer mHitgroupRecordsBuffer;

		/*! @{ Shader binding table */
		OptixShaderBindingTable mShaderBindingTable = {};
		/*! @} */
	};
}
