#include "OptixRenderer.h"

// Project
#include "Optix/OptixError.h"
#include "Renderer/Renderer.h"
#include "Utility/Errors.h"
#include "Utility/Logger.h"
#include "Utility/FileSystem.h"

// Optix
#include "optix7/optix.h"
#include "optix7/optix_stubs.h"
#include "optix7/optix_function_table.h"
#include "optix7/optix_function_table_definition.h"
#include "optix7/optix_stack_size.h"

namespace Tracer
{
	namespace
	{
		// Raygen program Shader Binding Table record
		struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
		{
			char header[OPTIX_SBT_RECORD_HEADER_SIZE] = {};
			//void* data = nullptr;
		};



		// Miss program Shader Binding Table record
		struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
		{
			char header[OPTIX_SBT_RECORD_HEADER_SIZE] = {};
			//void* data = nullptr;
		};



		// Hitgroup program Shader Binding Table record
		struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
		{
			char header[OPTIX_SBT_RECORD_HEADER_SIZE] = {};
			//void* data = nullptr;
		};



		void OptixLogCallback(unsigned int level, const char* tag, const char* message, void* /*cbdata*/) noexcept
		{
			Logger::Info("[OptiX] %d - %s: %s", level, tag, message);
		}



		void InitOptix()
		{
			static bool isInitialized = false;
			if(!isInitialized)
			{
				OPTIX_CHECK(optixInit());
				isInitialized = true;
			}
		}
	}



	OptixRenderer::OptixRenderer(CUcontext cudaContext)
	{
		InitOptix();

		OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &mDeviceContext));
		OPTIX_CHECK(optixDeviceContextSetLogCallback(mDeviceContext, OptixLogCallback, nullptr, 4));

		CreatePipeline();
		CreateShaderBindingTable();
	}



	OptixRenderer::~OptixRenderer()
	{
		OPTIX_CHECK(optixModuleDestroy(mModule));
		OPTIX_CHECK(optixPipelineDestroy(mPipeline));
		OPTIX_CHECK(optixDeviceContextDestroy(mDeviceContext));
	}



	void OptixRenderer::BuildAccel(const std::vector<OptixInstance>& instances)
	{
		if(instances.size() == 0)
		{
			mSceneRoot = 0;
			mInstancesBuffer.Free();
			return;
		}

		// upload instances
		mInstancesBuffer.UploadAsync(instances, true);

		// build top-level
		OptixBuildInput instanceBuildInput = {};
		instanceBuildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		instanceBuildInput.instanceArray.instances    = mInstancesBuffer.DevicePtr();
		instanceBuildInput.instanceArray.numInstances = static_cast<uint32_t>(instances.size());

		// Acceleration setup
		OptixAccelBuildOptions buildOptions = {};
		buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
		buildOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

		OptixAccelBufferSizes accelBufferSizes = {};
		OPTIX_CHECK(optixAccelComputeMemoryUsage(mDeviceContext, &buildOptions, &instanceBuildInput, 1, &accelBufferSizes));

		// Execute build
		CudaBuffer tempBuffer(accelBufferSizes.tempSizeInBytes);
		mAccelBuffer.Resize(accelBufferSizes.outputSizeInBytes);
		CUDA_CHECK(cudaDeviceSynchronize());
		OPTIX_CHECK(optixAccelBuild(mDeviceContext, nullptr, &buildOptions, &instanceBuildInput, 1,
									tempBuffer.DevicePtr(), tempBuffer.Size(), mAccelBuffer.DevicePtr(), mAccelBuffer.Size(),
									&mSceneRoot, nullptr, 0));
	}



	void OptixRenderer::TraceRays(CUstream stream, CudaBuffer& launchParams, uint32_t width, uint32_t height, uint32_t depth)
	{
		OPTIX_CHECK(optixLaunch(mPipeline, stream, launchParams.DevicePtr(), launchParams.Size(),
								&mShaderBindingTable, width, height, depth));
	}



	void OptixRenderer::CreatePipeline()
	{
		// module compile options
		OptixModuleCompileOptions moduleCompileOptions = {};
		moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef _DEBUG
		moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
		moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
		moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
		moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#endif

		// pipeline compile options
		OptixPipelineCompileOptions pipelineCompileOptions = {};
		pipelineCompileOptions.usesMotionBlur                   = false;
		pipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
		pipelineCompileOptions.numPayloadValues                 = 4;
		pipelineCompileOptions.numAttributeValues               = 2;
		pipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_DEBUG;
		pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
		pipelineCompileOptions.usesPrimitiveTypeFlags           = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

		// pipeline link options
		OptixPipelineLinkOptions pipelineLinkOptions = {};
		pipelineLinkOptions.maxTraceDepth          = 1;
		pipelineLinkOptions.debugLevel             = moduleCompileOptions.debugLevel;

		// load PTX
#ifdef _DEBUG
		std::vector<char> ptxCode = ReadBinaryFile("optix_debug.optixir");
		assert(!ptxCode.empty());
#else
		std::vector<char> ptxCode = ReadBinaryFile("optix.optixir");
#endif

		char log[1u << 16];
		size_t sizeof_log = sizeof(log);
		const OptixResult moduleCreateResult = optixModuleCreateFromPTX(mDeviceContext, &moduleCompileOptions,
			&pipelineCompileOptions, ptxCode.data(), ptxCode.size(), log, &sizeof_log, &mModule);
		if(moduleCreateResult != OPTIX_SUCCESS)
			FatalError(log);

			if(sizeof_log > 1)
				Logger::Info("%s", log);

		// ray gen
		OptixProgramGroupDesc raygenDesc      = {};
		raygenDesc.kind                       = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		raygenDesc.raygen.module              = mModule;
		raygenDesc.raygen.entryFunctionName   = "__raygen__";
		mRayGenProgram                        = CreateProgram({}, raygenDesc);

		// miss
		OptixProgramGroupDesc missDesc        = {};
		missDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_MISS;
		missDesc.miss.module                  = mModule;
		missDesc.miss.entryFunctionName       = "__miss__";
		mMissProgram                          = CreateProgram({}, missDesc);

		// hit
		OptixProgramGroupDesc hitDesc         = {};
		hitDesc.kind                          = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitDesc.hitgroup.moduleCH             = mModule;
		hitDesc.hitgroup.entryFunctionNameCH  = "__closesthit__";
		hitDesc.hitgroup.moduleAH             = mModule;
		hitDesc.hitgroup.entryFunctionNameAH  = "__anyhit__";
		mHitgroupProgram                      = CreateProgram({}, hitDesc);

		// group programs
		std::vector<OptixProgramGroup> programGroups =
		{
			mRayGenProgram,
			mMissProgram,
			mHitgroupProgram
		};

		// create pipeline
		OPTIX_CHECK(optixPipelineCreate(mDeviceContext, &pipelineCompileOptions, &pipelineLinkOptions,
														 programGroups.data(), static_cast<unsigned int>(programGroups.size()),
														 log, &sizeof_log, &mPipeline));
		if(sizeof_log > 1)
			Logger::Info("%s", log);

		// set stack sizes
		OptixStackSizes stackSizes = {};
		for(OptixProgramGroup& g : programGroups)
			OPTIX_CHECK(optixUtilAccumulateStackSizes(g, &stackSizes));

		uint32_t directCallableStackSizeFromTraversal = 2 << 10;
		uint32_t directCallableStackSizeFromState     = 2 << 10;
		uint32_t continuationStackSize                = 2 << 10;
		uint32_t maxTraversableGraphDepth             = 3;
		OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes, Renderer::MaxTraceDepth, 0, 0, &directCallableStackSizeFromTraversal,
											   &directCallableStackSizeFromState, &continuationStackSize));
		OPTIX_CHECK(optixPipelineSetStackSize(mPipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState,
											  continuationStackSize, maxTraversableGraphDepth));
	}



	void OptixRenderer::CreateShaderBindingTable()
	{
		// raygen records
		RaygenRecord raygen = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(mRayGenProgram, &raygen));
		mRayGenRecordsBuffer.Upload(&raygen, 1, true);
		mShaderBindingTable.raygenRecord = mRayGenRecordsBuffer.DevicePtr();

		// miss records
		MissRecord miss = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(mMissProgram, &miss));
		mMissRecordsBuffer.Upload(&miss, 1, true);
		mShaderBindingTable.missRecordBase          = mMissRecordsBuffer.DevicePtr();
		mShaderBindingTable.missRecordStrideInBytes = sizeof(MissRecord);
		mShaderBindingTable.missRecordCount         = 1;

		// hitgroup records
		HitgroupRecord hit = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(mHitgroupProgram, &hit));
		mHitRecordsBuffer.Upload(&hit, 1, true);
		mShaderBindingTable.hitgroupRecordBase          = mHitRecordsBuffer.DevicePtr();
		mShaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
		mShaderBindingTable.hitgroupRecordCount         = 1;
	}



	OptixProgramGroup OptixRenderer::CreateProgram(const OptixProgramGroupOptions& options, const OptixProgramGroupDesc& desc)
	{
		char log[2048];
		size_t logLength = sizeof(log);
		OptixProgramGroup program {};
		OPTIX_CHECK(optixProgramGroupCreate(mDeviceContext, &desc, 1, &options, log, &logLength, &program));
		if(logLength > 1)
			Logger::Info("%s", log);
		return program;
	}
}
