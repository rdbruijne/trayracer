#pragma once

// Project
#include "CUDA/CudaBuffer.h"
#include "Common/CommonStructs.h"

// C++
#include <string>
#include <vector>

namespace Tracer
{
	class Renderer
	{
	public:
		explicit Renderer(const int2& resolution);
		~Renderer();

		void RenderFrame();

		void DownloadPixels(std::vector<uint32_t>& dstPixels);

		void Resize(const int2& resolution);
		void SetSceneRoot(OptixTraversableHandle sceneRoot);
		void SetCamera(float3 cameraPos, float3 cameraForward, float3 cameraUp, float camFov);

		inline OptixDeviceContext GetOptixDeviceContext() const
		{
			return mOptixContext;
		}

	private:
		// Creation
		void CreateContext();
		void CreateModule();
		void CreateRaygenPrograms();
		void CreateMissPrograms();
		void CreateHitgroupPrograms();
		void CreatePipeline();
		void BuildShaderBindingTable();

		// Render buffer
		CudaBuffer mColorBuffer;

		// Launch parameters
		LaunchParams mLaunchParams;
		// CUDA buffer for the launch parameters
		CudaBuffer   mLaunchParamsBuffer;

		// CUDA device context
		CUcontext      mCudaContext;
		// CUDA stream
		CUstream       mStream;
		// CUDA device properties
		cudaDeviceProp mDeviceProperties;

		// OptiX module
		OptixModule               mModule;
		// Compile options for mModule
		OptixModuleCompileOptions mModuleCompileOptions;

		// OptiX pipeline
		OptixPipeline               mPipeline;
		// Compile options for mPipeline
		OptixPipelineCompileOptions mPipelineCompileOptions;
		// Link options for mPipeline
		OptixPipelineLinkOptions    mPipelineLinkOptions;

		// The OptiX device context
		OptixDeviceContext mOptixContext;

		// Ray generation programs
		std::vector<OptixProgramGroup> mRayGenPrograms;
		// CUDA buffer for mRayGenPrograms
		CudaBuffer mRaygenRecordsBuffer;

		// Miss programs
		std::vector<OptixProgramGroup> mMissPrograms;
		// CUDA buffer for mMissPrograms
		CudaBuffer mMissRecordsBuffer;

		// Hit programs
		std::vector<OptixProgramGroup> mHitgroupPrograms;
		// CUDA buffer for mHitgroupPrograms
		CudaBuffer mHitgroupRecordsBuffer;

		// Shader binding table
		OptixShaderBindingTable mShaderBindingTable = {};
	};
}
