#pragma once

// Project
#include "CUDA/CudaBuffer.h"

// Optix
#include "optix7/optix.h"

namespace Tracer
{
	class OptixRenderer
	{
	public:
		explicit OptixRenderer(CUcontext cudaContext);
		~OptixRenderer();

		OptixModule Module() { return mModule; }
		OptixPipeline Pipeline() { return mPipeline; }
		OptixDeviceContext DeviceContext() { return mDeviceContext; }

		OptixTraversableHandle SceneRoot() const { return mSceneRoot; }

		void BuildAccel(const std::vector<OptixInstance>& instances);
		void TraceRays(CUstream stream, CudaBuffer& launchParams, uint32_t width, uint32_t height, uint32_t depth);

	private:
		void CreatePipeline();
		void CreateShaderBindingTable();

		OptixProgramGroup CreateProgram(const OptixProgramGroupOptions& options, const OptixProgramGroupDesc& desc);

		OptixModule mModule = nullptr;
		OptixPipeline mPipeline = nullptr;
		OptixDeviceContext mDeviceContext = nullptr;

		// Programs
		OptixProgramGroup mRayGenProgram = nullptr;
		OptixProgramGroup mMissProgram = nullptr;
		OptixProgramGroup mHitgroupProgram = nullptr;

		// Records
		CudaBuffer mRayGenRecordsBuffer = {};
		CudaBuffer mMissRecordsBuffer = {};
		CudaBuffer mHitRecordsBuffer = {};

		// scene
		OptixTraversableHandle mSceneRoot = 0;
		CudaBuffer mAccelBuffer = {};
		CudaBuffer mInstancesBuffer = {};

		// shader binding table
		OptixShaderBindingTable mShaderBindingTable = {};
	};
}
