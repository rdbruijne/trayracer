#pragma once

// Project
#include "CUDA/CudaBuffer.h"
#include "Common/CommonStructs.h"

// C++
#include <string>
#include <vector>

namespace Tracer
{
	class Scene;
	class Renderer
	{
	public:
		explicit Renderer(const int2& resolution);
		~Renderer();

		void BuildScene(Scene* scene);
		void RenderFrame();

		void DownloadPixels(std::vector<uint32_t>& dstPixels);

		void Resize(const int2& resolution);
		void SetCamera(float3 cameraPos, float3 cameraForward, float3 cameraUp, float camFov);

	private:
		// Creation
		void CreateContext();
		void CreateModule();
		void CreateRaygenPrograms();
		void CreateMissPrograms();
		void CreateHitgroupPrograms();
		void CreatePipeline();

		// scene building
		void BuildGeometry(Scene* scene);
		void BuildShaderBindingTable(Scene* scene);

		// Render buffer
		CudaBuffer mColorBuffer;

		// Launch parameters
		LaunchParams mLaunchParams;
		CudaBuffer mLaunchParamsBuffer;

		// CUDA device properties
		CUcontext mCudaContext;
		CUstream mStream;
		cudaDeviceProp mDeviceProperties;

		// OptiX module
		OptixModule mModule;
		OptixModuleCompileOptions mModuleCompileOptions;

		// OptiX pipeline
		OptixPipeline mPipeline;
		OptixPipelineCompileOptions mPipelineCompileOptions;
		OptixPipelineLinkOptions mPipelineLinkOptions;

		// The OptiX device context
		OptixDeviceContext mOptixContext;

		// Ray generation programs
		std::vector<OptixProgramGroup> mRayGenPrograms;
		CudaBuffer mRaygenRecordsBuffer;

		// Miss programs
		std::vector<OptixProgramGroup> mMissPrograms;
		CudaBuffer mMissRecordsBuffer;

		// Hit programs
		std::vector<OptixProgramGroup> mHitgroupPrograms;
		CudaBuffer mHitgroupRecordsBuffer;

		// Shader binding table
		OptixShaderBindingTable mShaderBindingTable = {};

		// Geometry
		OptixTraversableHandle mSceneRoot = 0;
		CudaBuffer mVertexBuffer;
		CudaBuffer mIndexBuffer;
		CudaBuffer mAccelBuffer;
	};
}
