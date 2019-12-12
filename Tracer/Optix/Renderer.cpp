#include "Renderer.h"

// C++
#include <assert.h>

namespace Tracer
{
	namespace
	{
		/*! Raygen program Shader Binding Table record */
		struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
		{
			char header[OPTIX_SBT_RECORD_HEADER_SIZE] = {};
			void* data = nullptr;
		};



		/*! Miss program Shader Binding Table record */
		struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
		{
			char header[OPTIX_SBT_RECORD_HEADER_SIZE] = {};
			void* data = nullptr;
		};



		/*! Hitgroup program Shader Binding Table record */
		struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
		{
			char header[OPTIX_SBT_RECORD_HEADER_SIZE] = {};
			int objectID = 0;
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
		BuildShaderBindingTable();

		// resize buffer
		Resize(resolution);

		// allocate launch params
		mLaunchParamsBuffer.Alloc(sizeof(LaunchParams));
	}



	Renderer::~Renderer()
	{
	}



	void Renderer::RenderFrame()
	{
		if(mLaunchParams.resolutionX == 0 || mLaunchParams.resolutionY == 0)
			return;

		mLaunchParamsBuffer.Upload(&mLaunchParams, 1);
		mLaunchParams.frameID++;

		optixLaunch(mPipeline, mStream, mLaunchParamsBuffer.DevicePtr(), mLaunchParamsBuffer.Size(), &mShaderBindingTable,
					static_cast<unsigned int>(mLaunchParams.resolutionX), static_cast<unsigned int>(mLaunchParams.resolutionY), 1);
		cudaDeviceSynchronize();
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



	void Renderer::CreateContext()
	{
		const int deviceID = 0;
		cudaSetDevice(deviceID);
		cudaStreamCreate(&mStream);

		cudaGetDeviceProperties(&mDeviceProperties, deviceID);
		printf("Running on %s\n", mDeviceProperties.name);

		const CUresult cuRes = cuCtxGetCurrent(&mCudaContext);
		assert(cuRes == CUDA_SUCCESS);

		optixDeviceContextCreate(mCudaContext, 0, &mOptixContext);
		optixDeviceContextSetLogCallback(mOptixContext, OptixLogCallback, nullptr, 4);
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
		const OptixResult optixRes = optixModuleCreateFromPTX(mOptixContext, &mModuleCompileOptions, &mPipelineCompileOptions,
															  ptxCode.c_str(), ptxCode.size(), log, &sizeof_log, &mModule);
		assert(optixRes == OPTIX_SUCCESS);

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
		const OptixResult optixRes = optixProgramGroupCreate(mOptixContext, &desc, 1, &options, log, &sizeof_log, &mRayGenPrograms[0]);
		assert(optixRes == OPTIX_SUCCESS);

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
		const OptixResult optixRes = optixProgramGroupCreate(mOptixContext, &desc, 1, &options, log, &logLength, &mMissPrograms[0]);
		assert(optixRes == OPTIX_SUCCESS);

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
		const OptixResult optixRes = optixProgramGroupCreate(mOptixContext, &desc, 1, &options, log, &logLength, &mHitgroupPrograms[0]);
		assert(optixRes == OPTIX_SUCCESS);

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
		const OptixResult optixRes = optixPipelineCreate(mOptixContext, &mPipelineCompileOptions, &mPipelineLinkOptions,
														 programGroups.data(), static_cast<unsigned int>(programGroups.size()),
														 log, &logLength, &mPipeline);
		assert(optixRes == OPTIX_SUCCESS);

		if (logLength > 1)
			printf("%s\n", log);

		const OptixResult optixRes2 = optixPipelineSetStackSize(mPipeline,
																2 << 10, // directCallableStackSizeFromTraversal
																2 << 10, // directCallableStackSizeFromState
																2 << 10, // continuationStackSize
																3);      // maxTraversableGraphDepth
		assert(optixRes2 == OPTIX_SUCCESS);
	}



	void Renderer::BuildShaderBindingTable()
	{
		// raygen records
		std::vector<RaygenRecord> raygenRecords;
		raygenRecords.reserve(mRayGenPrograms.size());
		for(auto p : mRayGenPrograms)
		{
			RaygenRecord r;
			optixSbtRecordPackHeader(p, &r);
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
			optixSbtRecordPackHeader(p, &r);
			r.data = nullptr;
			missRecords.push_back(r);
		}
		mMissRecordsBuffer.AllocAndUpload(missRecords);
		mShaderBindingTable.missRecordBase          = mMissRecordsBuffer.DevicePtr();
		mShaderBindingTable.missRecordStrideInBytes = sizeof(MissRecord);
		mShaderBindingTable.missRecordCount         = static_cast<unsigned int>(missRecords.size());

		// hitgroup records
		const int numObjects = 1; // dummy object to prevent nullptr.
		std::vector<HitgroupRecord> hitgroupRecords;
		hitgroupRecords.reserve(numObjects);
		for(int i = 0; i < numObjects; i++)
		{
			uint64_t objectType = 0;
			HitgroupRecord r;
			optixSbtRecordPackHeader(mHitgroupPrograms[objectType], &r);
			r.objectID = i;
			hitgroupRecords.push_back(r);
		}
		mHitgroupRecordsBuffer.AllocAndUpload(hitgroupRecords);
		mShaderBindingTable.hitgroupRecordBase          = mHitgroupRecordsBuffer.DevicePtr();
		mShaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
		mShaderBindingTable.hitgroupRecordCount         = static_cast<unsigned int>(hitgroupRecords.size());
	}
}
