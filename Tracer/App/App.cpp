#include "App.h"

// Project
#include "OptixHelpers.h"
#include "Renderer.h"

namespace Tracer
{
	void App::Init(Renderer* renderer, Window* window)
	{
		CreateScene();
		BuildScene(renderer);
	}



	void App::DeInit(Renderer* renderer, Window* window)
	{
	}



	void App::Tick(Renderer* renderer, Window* window, float dt /*= 1.f / 60.f*/)
	{
		renderer->SetSceneRoot(mSceneRoot);
		//renderer->SetCamera(make_float3(0, 0, -10), make_float3(0, 0, 0), make_float3(0, 1, 0));
		renderer->SetCamera(make_float3(-10, 10, -12), make_float3(0, 0, 0), make_float3(0, 1, 0));
	}



	void App::CreateScene()
	{
		// cube
		mMesh = std::make_shared<Mesh>(
			std::vector<float3>
			{
				make_float3(-0.5f, -0.5f, -0.5f),
				make_float3(-0.5f, -0.5f,  0.5f),
				make_float3(-0.5f,  0.5f, -0.5f),
				make_float3(-0.5f,  0.5f,  0.5f),
				make_float3( 0.5f, -0.5f, -0.5f),
				make_float3( 0.5f, -0.5f,  0.5f),
				make_float3( 0.5f,  0.5f, -0.5f),
				make_float3( 0.5f,  0.5f,  0.5f)
			},
			std::vector<float3>(), // ignore normals
			std::vector<float3>(), // ignore texcoords
			std::vector<uint3>
			{
				make_uint3(0, 3, 2),
				make_uint3(0, 1, 3),
				make_uint3(4, 7, 5),
				make_uint3(4, 6, 7),
				make_uint3(0, 5, 1),
				make_uint3(0, 4, 5),
				make_uint3(2, 7, 6),
				make_uint3(2, 3, 7),
				make_uint3(0, 6, 4),
				make_uint3(0, 2, 6),
				make_uint3(1, 7, 3),
				make_uint3(1, 5, 7)
			});
	}



	void App::BuildScene(Renderer* renderer)
	{
		//--------------------------------
		// Build input
		//--------------------------------
		OptixBuildInput buildInput = {};
		buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		// vertices
		mVertexBuffer.AllocAndUpload(mMesh->GetVertices());
		buildInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
		buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
		buildInput.triangleArray.numVertices         = static_cast<unsigned int>(mMesh->GetVertices().size());
		buildInput.triangleArray.vertexBuffers       = mVertexBuffer.DevicePtrPtr();

		// indices
		mIndexBuffer.AllocAndUpload(mMesh->GetIndices());
		buildInput.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
		buildInput.triangleArray.numIndexTriplets   = static_cast<unsigned int>(mMesh->GetIndices().size());
		buildInput.triangleArray.indexBuffer        = mIndexBuffer.DevicePtr();

		// other
		const uint32_t buildFlags[] = { 0 };
		buildInput.triangleArray.flags                       = buildFlags;
		buildInput.triangleArray.numSbtRecords               = 1;
		buildInput.triangleArray.sbtIndexOffsetBuffer        = 0;
		buildInput.triangleArray.sbtIndexOffsetSizeInBytes   = 0;
		buildInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

		//--------------------------------
		// Acceleration setup
		//--------------------------------
		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags            = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accelOptions.motionOptions.numKeys = 1;
		accelOptions.operation             = OPTIX_BUILD_OPERATION_BUILD;

		OptixAccelBufferSizes accelBufferSizes = {};
		OPTIX_CHECK(optixAccelComputeMemoryUsage(renderer->GetOptixDeviceContext(), &accelOptions, &buildInput, 1, &accelBufferSizes));

		//--------------------------------
		// Prepare for compacting
		//--------------------------------
		CudaBuffer compactedSizeBuffer(sizeof(uint64_t));

		OptixAccelEmitDesc emitDesc;
		emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = compactedSizeBuffer.DevicePtr();

		//--------------------------------
		// Execute build
		//--------------------------------
		CudaBuffer tempBuffer(accelBufferSizes.tempSizeInBytes);
		CudaBuffer outputBuffer(accelBufferSizes.outputSizeInBytes);
		OPTIX_CHECK(optixAccelBuild(renderer->GetOptixDeviceContext(), nullptr,
									&accelOptions,
									&buildInput, 1,
									tempBuffer.DevicePtr(), tempBuffer.Size(),
									outputBuffer.DevicePtr(), outputBuffer.Size(),
									&mSceneRoot,
									&emitDesc, 1));
		CUDA_CHECK(cudaDeviceSynchronize());

		//--------------------------------
		// Compact
		//--------------------------------
		uint64_t compactedSize = 0;
		compactedSizeBuffer.Download(&compactedSize, 1);

		mAccelBuffer.Alloc(compactedSize);
		OPTIX_CHECK(optixAccelCompact(renderer->GetOptixDeviceContext(), nullptr, mSceneRoot, mAccelBuffer.DevicePtr(), mAccelBuffer.Size(), &mSceneRoot));
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}
