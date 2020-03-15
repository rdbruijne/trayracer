#include "Optix/Renderer.h"

// Project
#include "Optix/OptixHelpers.h"
#include "Resources/Scene.h"
#include "Utility/LinearMath.h"
#include "Utility/Utility.h"

// C++
#include <assert.h>

namespace Tracer
{
	namespace
	{
		// Raygen program Shader Binding Table record
		struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
		{
			char header[OPTIX_SBT_RECORD_HEADER_SIZE] = {};
			void* data = nullptr;
		};



		// Miss program Shader Binding Table record
		struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
		{
			char header[OPTIX_SBT_RECORD_HEADER_SIZE] = {};
			void* data = nullptr;
		};



		// Hitgroup program Shader Binding Table record
		struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
		{
			char header[OPTIX_SBT_RECORD_HEADER_SIZE] = {};
			TriangleMeshData meshData;
		};



		void OptixLogCallback(unsigned int level, const char* tag, const char* message, void* cbdata) noexcept
		{
			printf("[%1u][%-12s]: %s\n", level, tag, message);
		}
	}



	Renderer::Renderer(const int2& resolution)
	{
		// create context
		CreateContext();

		// create module
		CreateModule();

		// create programs
		CreateRaygenPrograms();
		CreateMissPrograms();
		CreateHitgroupPrograms();

		// create pipeline
		CreatePipeline();

		// build shader binding table
		BuildShaderBindingTable(nullptr);

		// resize buffer
		Resize(resolution);

		// allocate launch params
		mLaunchParamsBuffer.Alloc(sizeof(LaunchParams));
	}



	Renderer::~Renderer()
	{
	}



	void Renderer::BuildScene(Scene* scene)
	{
		BuildGeometry(scene);
		BuildShaderBindingTable(scene);
	}



	void Renderer::RenderFrame()
	{
		if(mLaunchParams.resolutionX == 0 || mLaunchParams.resolutionY == 0)
			return;

		// update launch params
		mLaunchParams.sceneRoot = mSceneRoot;
		mLaunchParams.frameID++;
		mLaunchParamsBuffer.Upload(&mLaunchParams, 1);

		// launch OptiX
		OPTIX_CHECK(optixLaunch(mPipeline, mStream, mLaunchParamsBuffer.DevicePtr(), mLaunchParamsBuffer.Size(), &mShaderBindingTable,
								static_cast<unsigned int>(mLaunchParams.resolutionX), static_cast<unsigned int>(mLaunchParams.resolutionY), 1));
		CUDA_CHECK(cudaDeviceSynchronize());
	}



	void Renderer::DownloadPixels(std::vector<uint32_t>& dstPixels)
	{
		dstPixels.resize(static_cast<size_t>(mLaunchParams.resolutionX) * mLaunchParams.resolutionY);
		mColorBuffer.Download(dstPixels.data(), static_cast<size_t>(mLaunchParams.resolutionX) * mLaunchParams.resolutionY);
	}



	void Renderer::Resize(const int2& resolution)
	{
		// resize buffer
		mColorBuffer.Resize(sizeof(uint32_t) * resolution.x * resolution.y);

		// update launch params
		mLaunchParams.resolutionX = resolution.x;
		mLaunchParams.resolutionY = resolution.y;
		mLaunchParams.colorBuffer = reinterpret_cast<uint32_t*>(mColorBuffer.DevicePtr());
	}



	void Renderer::SetCamera(float3 cameraPos, float3 cameraForward, float3 cameraUp, float camFov)
	{
		mLaunchParams.cameraFov     = camFov;
		mLaunchParams.cameraPos     = cameraPos;
		mLaunchParams.cameraForward = cameraForward;
		mLaunchParams.cameraSide    = normalize(cross(cameraForward, cameraUp));
		mLaunchParams.cameraUp      = normalize(cross(mLaunchParams.cameraSide, cameraForward));
	}



	void Renderer::CreateContext()
	{
		const int deviceID = 0;
		CUDA_CHECK(cudaSetDevice(deviceID));
		CUDA_CHECK(cudaStreamCreate(&mStream));

		CUDA_CHECK(cudaGetDeviceProperties(&mDeviceProperties, deviceID));
		printf("Running on %s\n", mDeviceProperties.name);

		CU_CHECK(cuCtxGetCurrent(&mCudaContext));

		OPTIX_CHECK(optixDeviceContextCreate(mCudaContext, 0, &mOptixContext));
		OPTIX_CHECK(optixDeviceContextSetLogCallback(mOptixContext, OptixLogCallback, nullptr, 4));
	}



	void Renderer::CreateModule()
	{
		// module compile options
		mModuleCompileOptions.maxRegisterCount = 100;
#ifdef _DEBUG
		mModuleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
		mModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
		mModuleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
		mModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

		// pipeline compile options
		mPipelineCompileOptions                                  = {};
		mPipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
		mPipelineCompileOptions.usesMotionBlur                   = false;
		mPipelineCompileOptions.numPayloadValues                 = 2;
		mPipelineCompileOptions.numAttributeValues               = 2;
		mPipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
		mPipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

		// pipeline link options
		mPipelineLinkOptions.overrideUsesMotionBlur = false;
		mPipelineLinkOptions.maxTraceDepth          = 2;

		// load PTX
		const std::string ptxCode = ReadFile("ptx/program.ptx");
		assert(!ptxCode.empty());

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixModuleCreateFromPTX(mOptixContext, &mModuleCompileOptions, &mPipelineCompileOptions,
											 ptxCode.c_str(), ptxCode.size(), log, &sizeof_log, &mModule));
		if (sizeof_log > 1)
			printf("%s\n", log);
	}



	void Renderer::CreateRaygenPrograms()
	{
		mRayGenPrograms.resize(1);

		OptixProgramGroupOptions options = {};
		OptixProgramGroupDesc desc       = {};
		desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		desc.raygen.module               = mModule;
		desc.raygen.entryFunctionName    = "__raygen__renderFrame";

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(mOptixContext, &desc, 1, &options, log, &sizeof_log, &mRayGenPrograms[0]));
		if (sizeof_log > 1)
			printf("%s\n", log);
	}



	void Renderer::CreateMissPrograms()
	{
		mMissPrograms.resize(1);

		OptixProgramGroupOptions options = {};
		OptixProgramGroupDesc desc       = {};
		desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_MISS;
		desc.miss.module                 = mModule;
		desc.miss.entryFunctionName      = "__miss__radiance";

		char log[2048];
		size_t logLength = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(mOptixContext, &desc, 1, &options, log, &logLength, &mMissPrograms[0]));
		if (logLength > 1)
			printf("%s\n", log);
	}



	void Renderer::CreateHitgroupPrograms()
	{
		mHitgroupPrograms.resize(1);

		OptixProgramGroupOptions options  = {};
		OptixProgramGroupDesc desc        = {};
		desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		desc.hitgroup.moduleCH            = mModule;
		desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
		desc.hitgroup.moduleAH            = mModule;
		desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

		char log[2048];
		size_t logLength = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(mOptixContext, &desc, 1, &options, log, &logLength, &mHitgroupPrograms[0]));
		if (logLength > 1)
			printf("%s\n", log);
	}



	void Renderer::CreatePipeline()
	{
		std::vector<OptixProgramGroup> programGroups;
		programGroups.reserve(mRayGenPrograms.size() + mMissPrograms.size() + mHitgroupPrograms.size());
		for(auto p : mRayGenPrograms)
			programGroups.push_back(p);
		for (auto p : mMissPrograms)
			programGroups.push_back(p);
		for (auto p : mHitgroupPrograms)
			programGroups.push_back(p);

		char log[2048];
		size_t logLength = sizeof(log);
		OPTIX_CHECK(optixPipelineCreate(mOptixContext, &mPipelineCompileOptions, &mPipelineLinkOptions,
														 programGroups.data(), static_cast<unsigned int>(programGroups.size()),
														 log, &logLength, &mPipeline));
		if (logLength > 1)
			printf("%s\n", log);

		OPTIX_CHECK(optixPipelineSetStackSize(mPipeline,
											  2 << 10, // directCallableStackSizeFromTraversal
											  2 << 10, // directCallableStackSizeFromState
											  2 << 10, // continuationStackSize
											  3));      // maxTraversableGraphDepth
	}



	void Renderer::BuildGeometry(Scene* scene)
	{
		// #TODO(RJCDB): support empty scenes
		assert(scene->mMeshes.size() != 0 && scene->mMaterials.size() != 0);

		//--------------------------------
		// Build input
		//--------------------------------
		OptixBuildInput buildInput = {};
		buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		// vertices
		mVertexBuffer.AllocAndUpload(scene->mMeshes[0]->GetVertices());
		buildInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
		buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
		buildInput.triangleArray.numVertices         = static_cast<unsigned int>(scene->mMeshes[0]->GetVertices().size());
		buildInput.triangleArray.vertexBuffers       = mVertexBuffer.DevicePtrPtr();

		// indices
		mIndexBuffer.AllocAndUpload(scene->mMeshes[0]->GetIndices());
		buildInput.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
		buildInput.triangleArray.numIndexTriplets   = static_cast<unsigned int>(scene->mMeshes[0]->GetIndices().size());
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
		OPTIX_CHECK(optixAccelComputeMemoryUsage(mOptixContext, &accelOptions, &buildInput, 1, &accelBufferSizes));

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
		OPTIX_CHECK(optixAccelBuild(mOptixContext, nullptr,
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
		OPTIX_CHECK(optixAccelCompact(mOptixContext, nullptr, mSceneRoot, mAccelBuffer.DevicePtr(), mAccelBuffer.Size(), &mSceneRoot));
		CUDA_CHECK(cudaDeviceSynchronize());
	}



	void Renderer::BuildShaderBindingTable(Scene* scene)
	{
		// raygen records
		std::vector<RaygenRecord> raygenRecords;
		raygenRecords.reserve(mRayGenPrograms.size());
		for(auto p : mRayGenPrograms)
		{
			RaygenRecord r;
			OPTIX_CHECK(optixSbtRecordPackHeader(p, &r));
			r.data = nullptr;
			raygenRecords.push_back(r);
		}
		mRaygenRecordsBuffer.AllocAndUpload(raygenRecords);
		mShaderBindingTable.raygenRecord = mRaygenRecordsBuffer.DevicePtr();

		// miss records
		std::vector<MissRecord> missRecords;
		missRecords.reserve(mMissPrograms.size());
		for(auto p : mMissPrograms)
		{
			MissRecord r;
			OPTIX_CHECK(optixSbtRecordPackHeader(p, &r));
			r.data = nullptr;
			missRecords.push_back(r);
		}
		mMissRecordsBuffer.AllocAndUpload(missRecords);
		mShaderBindingTable.missRecordBase          = mMissRecordsBuffer.DevicePtr();
		mShaderBindingTable.missRecordStrideInBytes = sizeof(MissRecord);
		mShaderBindingTable.missRecordCount         = static_cast<unsigned int>(missRecords.size());

		// hitgroup records
		const size_t numObjects = scene ? scene->mMaterials.size() : 0;
		std::vector<HitgroupRecord> hitgroupRecords;
		if(numObjects == 0)
		{
			// dummy material
			hitgroupRecords.reserve(1);
			uint64_t objectType = 0;
			HitgroupRecord r;
			OPTIX_CHECK(optixSbtRecordPackHeader(mHitgroupPrograms[objectType], &r));
			r.meshData.diffuse = make_float3(.75f, 0, .75f);
			hitgroupRecords.push_back(r);
		}
		else
		{
			hitgroupRecords.reserve(numObjects);
			for(size_t i = 0; i < numObjects; i++)
			{
				uint64_t objectType = 0;
				HitgroupRecord r;
				OPTIX_CHECK(optixSbtRecordPackHeader(mHitgroupPrograms[objectType], &r));
				r.meshData.diffuse = scene->mMaterials[i]->Diffuse;
				hitgroupRecords.push_back(r);
			}
		}
		mHitgroupRecordsBuffer.AllocAndUpload(hitgroupRecords);
		mShaderBindingTable.hitgroupRecordBase          = mHitgroupRecordsBuffer.DevicePtr();
		mShaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
		mShaderBindingTable.hitgroupRecordCount         = static_cast<unsigned int>(hitgroupRecords.size());
	}
}
