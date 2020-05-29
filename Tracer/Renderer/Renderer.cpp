#include "Renderer/Renderer.h"

// Project
#include "OpenGL/GLTexture.h"
#include "Renderer/Scene.h"
#include "Resources/CameraNode.h"
#include "Resources/Instance.h"
#include "Resources/Material.h"
#include "Resources/Model.h"
#include "Resources/Texture.h"
#include "Renderer/OptixError.h"
#include "Utility/LinearMath.h"
#include "Utility/Utility.h"

// SPT
#include "CUDA/CudaFwd.h"

// Optix
#pragma warning(push)
#pragma warning(disable: 4061 4365 5039 6011 6387 26451)
#include "optix7/optix.h"
#include "optix7/optix_stubs.h"
#include "optix7/optix_function_table.h"
#include "optix7/optix_function_table_definition.h"
#include "optix7/optix_stack_size.h"
#pragma warning(pop)

// CUDA
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

// C++
#include <assert.h>
#include <set>
#include <string>

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



		void OptixLogCallback(unsigned int level, const char* tag, const char* message, void* cbdata) noexcept
		{
			printf("[%1u][%-12s]: %s\n", level, tag, message);
		}
	}



	Renderer::Renderer()
	{
		// create Optix content
		CreateContext();
		CreateModule();
		CreatePrograms();
		CreatePipeline();
		CreateDenoiser();

		// build shader binding table
		CreateShaderBindingTable();

		// allocate launch params
		mLaunchParamsBuffer.Alloc(sizeof(LaunchParams));

		// set launch param constants
		mLaunchParams.multiSample  = 1;
		mLaunchParams.maxDepth     = 16;
		mLaunchParams.epsilon      = Epsilon;
		mLaunchParams.aoDist       = 1.f;
		mLaunchParams.zDepthMax    = 1.f;
		mLaunchParams.skyColor     = make_float3(1);
	}



	Renderer::~Renderer()
	{
		// #TODO: proper cleanup
		if(mCudaGraphicsResource)
			CUDA_CHECK(cudaGraphicsUnregisterResource(mCudaGraphicsResource));
	}



	void Renderer::BuildScene(Scene* scene)
	{
		// #TODO: async?
		BuildGeometry(scene);
		BuildMaterials(scene);

		mLaunchParams.sceneRoot = mSceneRoot;
		mLaunchParams.sampleCount = 0;
	}



	void Renderer::RenderFrame(GLTexture* renderTexture)
	{
		const int2 texRes = renderTexture->Resolution();
		if(texRes.x == 0 || texRes.y == 0)
			return;

		if(mLaunchParams.resX != texRes.x || mLaunchParams.resY != texRes.y)
			Resize(texRes);

		// prepare SPT buffers
		const uint32_t stride = mLaunchParams.resX * mLaunchParams.resY * mLaunchParams.multiSample;
		if(mPathStates.Size() != sizeof(float4) * stride * 3)
		{
			mPathStates.Resize(sizeof(float4) * stride * 3);
			mLaunchParams.pathStates = mPathStates.Ptr<float4>();
		}

		if(mHitData.Size() != sizeof(uint4) * stride)
		{
			mHitData.Resize(sizeof(uint4) * stride);
			mLaunchParams.hitData = mHitData.Ptr<uint4>();
		}

		if(mShadowRayData.Size() != sizeof(float4) * stride * 3)
		{
			mShadowRayData.Resize(sizeof(float4) * stride * 3);
			mLaunchParams.shadowRays = mShadowRayData.Ptr<float4>();
		}

		// update counters
		Counters counters = {};
		if(mCountersBuffer.Size() == 0)
		{
			mCountersBuffer.Upload(&counters, 1, true);
			SetCudaCounters(mCountersBuffer.Ptr<Counters>());
		}

		// update launch params
		mLaunchParams.rayGenMode = RayGen_Primary;
		mLaunchParamsBuffer.Upload(&mLaunchParams);
		SetCudaLaunchParams(mLaunchParamsBuffer.Ptr<LaunchParams>());

		// reset stats
		mRenderStats = {};

		// loop
		uint32_t pathCount = stride;
		mRenderTimeEvents.Start(mStream);
		for(int pathLength = 0; pathLength < mLaunchParams.maxDepth; pathLength++)
		{
			// launch Optix
			mTraceTimeEvents[pathLength].Start(mStream);
			InitCudaCounters();
			if(pathLength == 0)
			{
				// primary
				mRenderStats.primaryPathCount = pathCount;
				OPTIX_CHECK(optixLaunch(mPipeline, mStream, mLaunchParamsBuffer.DevicePtr(), mLaunchParamsBuffer.Size(),
										&mShaderBindingTable, static_cast<unsigned int>(mLaunchParams.resX),
										static_cast<unsigned int>(mLaunchParams.resY), static_cast<unsigned int>(mLaunchParams.multiSample)));
			}
			else if(pathCount > 0)
			{
				// bounce
				mLaunchParams.rayGenMode = RayGen_Secondary;
				mLaunchParamsBuffer.Upload(&mLaunchParams);
				if(pathLength == 1)
					mRenderStats.secondaryPathCount = pathCount;
				else
					mRenderStats.deepPathCount += pathCount;
				OPTIX_CHECK(optixLaunch(mPipeline, mStream, mLaunchParamsBuffer.DevicePtr(), mLaunchParamsBuffer.Size(),
										&mShaderBindingTable, pathCount, 1, 1));
			}
			mRenderStats.pathCount += pathCount;
			mTraceTimeEvents[pathLength].Stop(mStream);

			// shade
			CUDA_CHECK(cudaDeviceSynchronize());
			mShadeTimeEvents[pathLength].Start(mStream);
			Shade(mRenderMode, pathCount, mAccumulator.Ptr<float4>(), mPathStates.Ptr<float4>(), mHitData.Ptr<uint4>(),
				  mShadowRayData.Ptr<float4>(), make_int2(mLaunchParams.resX, mLaunchParams.resY), stride, pathLength);
			mShadeTimeEvents[pathLength].Stop(mStream);

			// update counters
			CUDA_CHECK(cudaDeviceSynchronize());
			mCountersBuffer.Download(&counters, 1);
			pathCount = counters.extendRays;

			// shadow rays
			if(counters.shadowRays > 0)
			{
				// fire shadow rays
				mShadowTimeEvents[pathLength].Start(mStream);
				mLaunchParams.rayGenMode = RayGen_Shadow;
				mLaunchParamsBuffer.Upload(&mLaunchParams);
				OPTIX_CHECK(optixLaunch(mPipeline, mStream, mLaunchParamsBuffer.DevicePtr(), mLaunchParamsBuffer.Size(),
										&mShaderBindingTable, counters.shadowRays, 1, 1));
				mShadowTimeEvents[pathLength].Stop(mStream);

				// update stats
				mRenderStats.shadowRayCount += counters.shadowRays;
				mRenderStats.pathCount += counters.shadowRays;
				counters.shadowRays = 0;
			}
		}
		mRenderTimeEvents.Stop(mStream);

		// update sample count
		mLaunchParams.sampleCount += mLaunchParams.multiSample;

		// finalize the frame
		FinalizeFrame(mAccumulator.Ptr<float4>(), mColorBuffer.Ptr<float4>(), make_int2(mLaunchParams.resX, mLaunchParams.resY), mLaunchParams.sampleCount);

		// run denoiser
		if(ShouldDenoise())
		{
			mDenoiseTimeEvents.Start(mStream);

			// input
			OptixImage2D inputLayer;
			inputLayer.data = mColorBuffer.DevicePtr();
			inputLayer.width = mLaunchParams.resX;
			inputLayer.height = mLaunchParams.resY;
			inputLayer.rowStrideInBytes = mLaunchParams.resX * sizeof(float4);
			inputLayer.pixelStrideInBytes = sizeof(float4);
			inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

			// output
			OptixImage2D outputLayer;
			outputLayer.data = mDenoisedBuffer.DevicePtr();
			outputLayer.width = mLaunchParams.resX;
			outputLayer.height = mLaunchParams.resY;
			outputLayer.rowStrideInBytes = mLaunchParams.resX * sizeof(float4);
			outputLayer.pixelStrideInBytes = sizeof(float4);
			outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

			// denoise
			OptixDenoiserParams denoiserParams;
			denoiserParams.denoiseAlpha = 1;
			denoiserParams.hdrIntensity = 0;
			denoiserParams.blendFactor  = 1;

			OPTIX_CHECK(optixDenoiserInvoke(mDenoiser, 0, &denoiserParams, mDenoiserState.DevicePtr(), mDenoiserState.Size(),
											&inputLayer, 1, 0, 0, &outputLayer,
											mDenoiserScratch.DevicePtr(), mDenoiserScratch.Size()));
			mDenoiseTimeEvents.Stop(mStream);

			mDenoisedFrame = true;
		}
		else
		{
			// empty timing so that we can call "Elapsed"
			mDenoiseTimeEvents.Start(mStream);
			mDenoiseTimeEvents.Stop(mStream);

			CUDA_CHECK(cudaMemcpy(mDenoisedBuffer.Ptr(), mColorBuffer.Ptr(), mColorBuffer.Size(), cudaMemcpyDeviceToDevice));
			mDenoisedFrame = false;
		}

		// update the target
		if(mRenderTarget != renderTexture)
		{
			if(mCudaGraphicsResource)
				CUDA_CHECK(cudaGraphicsUnregisterResource(mCudaGraphicsResource));

			CUDA_CHECK(cudaGraphicsGLRegisterImage(&mCudaGraphicsResource, renderTexture->ID(), GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
			mRenderTarget = renderTexture;
		}

		// copy to GL texture
		cudaArray* cudaTexPtr = nullptr;
		CUDA_CHECK(cudaGraphicsMapResources(1, &mCudaGraphicsResource, 0));
		CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cudaTexPtr, mCudaGraphicsResource, 0, 0));
		CUDA_CHECK(cudaMemcpy2DToArray(cudaTexPtr, 0, 0, mDenoisedBuffer.Ptr(), texRes.x * sizeof(float4), texRes.x * sizeof(float4), texRes.y, cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaGraphicsUnmapResources(1, &mCudaGraphicsResource, 0));
		CUDA_CHECK(cudaDeviceSynchronize());

		// update timings
		mRenderStats.primaryPathTimeMs = mTraceTimeEvents[0].Elapsed();
		mRenderStats.secondaryPathTimeMs = mTraceTimeEvents[1].Elapsed();
		for(int i = 2; i < mLaunchParams.maxDepth; i++)
			mRenderStats.deepPathTimeMs += mTraceTimeEvents[i].Elapsed();
		for(int i = 0; i < mLaunchParams.maxDepth; i++)
			mRenderStats.shadeTimeMs += mShadeTimeEvents[i].Elapsed();
		for(int i = 0; i < mLaunchParams.maxDepth; i++)
			mRenderStats.shadowTimeMs += mShadowTimeEvents[i].Elapsed();
		mRenderStats.renderTimeMs = mRenderTimeEvents.Elapsed();
		mRenderStats.denoiseTimeMs = mDenoiseTimeEvents.Elapsed();
	}



	void Renderer::SetCamera(CameraNode& camNode)
	{
		if(camNode.IsDirty())
		{
			SetCamera(camNode.Position(), normalize(camNode.Target() - camNode.Position()), camNode.Up(), camNode.Fov());
			camNode.MarkClean();
		}
	}



	void Renderer::SetCamera(const float3& cameraPos, const float3& cameraForward, const float3& cameraUp, float camFov)
	{
		mLaunchParams.cameraFov     = camFov;
		mLaunchParams.cameraPos     = cameraPos;
		mLaunchParams.cameraForward = cameraForward;
		mLaunchParams.cameraSide    = normalize(cross(cameraForward, cameraUp));
		mLaunchParams.cameraUp      = normalize(cross(mLaunchParams.cameraSide, cameraForward));

		mLaunchParams.sampleCount   = 0;
	}



	void Renderer::SetRenderMode(RenderModes mode)
	{
		if (mode != mRenderMode)
		{
			mRenderMode = mode;
			mLaunchParams.sampleCount = 0;
		}
	}



	RayPickResult Renderer::PickRay(int2 pixelIndex)
	{
		// allocate result buffer
		CudaBuffer resultBuffer;
		resultBuffer.Alloc(sizeof(RayPickResult));

		// set ray pick specific launch param options
		mLaunchParams.rayGenMode    = RayGen_RayPick;
		mLaunchParams.rayPickPixel  = pixelIndex;
		mLaunchParams.rayPickResult = resultBuffer.Ptr<RayPickResult>();

		// upload launch params
		mLaunchParamsBuffer.Upload(&mLaunchParams);

		// launch the kernel
		OPTIX_CHECK(optixLaunch(mPipeline, mStream, mLaunchParamsBuffer.DevicePtr(), mLaunchParamsBuffer.Size(), &mShaderBindingTable, 1, 1, 1));
		CUDA_CHECK(cudaDeviceSynchronize());

		// read the raypick result
		RayPickResult result;
		resultBuffer.Download(&result);

		return result;
	}



	void Renderer::Resize(const int2& resolution)
	{
		// resize buffers
		mAccumulator.Resize(sizeof(float4) * resolution.x * resolution.y);
		mColorBuffer.Resize(sizeof(float4) * resolution.x * resolution.y);
		mDenoisedBuffer.Resize(sizeof(float4) * resolution.x * resolution.y);

		// update launch params
		mLaunchParams.sampleCount = 0;
		mLaunchParams.resX = resolution.x;
		mLaunchParams.resY = resolution.y;
		mLaunchParams.accumulator = mAccumulator.Ptr<float4>();

		// allocate denoiser memory
		OptixDenoiserSizes denoiserReturnSizes;
		OPTIX_CHECK(optixDenoiserComputeMemoryResources(mDenoiser, resolution.x, resolution.y, &denoiserReturnSizes));
		mDenoiserScratch.Resize(denoiserReturnSizes.recommendedScratchSizeInBytes);
		mDenoiserState.Resize(denoiserReturnSizes.stateSizeInBytes);
		OPTIX_CHECK(optixDenoiserSetup(mDenoiser, 0, resolution.x, resolution.y, mDenoiserState.DevicePtr(), mDenoiserState.Size(),
									   mDenoiserScratch.DevicePtr(), mDenoiserScratch.Size()));
	}



	bool Renderer::ShouldDenoise() const
	{
		return mDenoisingEnabled && (mLaunchParams.sampleCount >= mDenoiserSampleTreshold) &&
			((mRenderMode == RenderModes::AmbientOcclusion) || (mRenderMode == RenderModes::PathTracing));
	}



	void Renderer::CreateContext()
	{
		OPTIX_CHECK(optixInit());

		constexpr int deviceID = 0;
		CUDA_CHECK(cudaSetDevice(deviceID));
		CUDA_CHECK(cudaStreamCreate(&mStream));

		CUDA_CHECK(cudaGetDeviceProperties(&mDeviceProperties, deviceID));
		printf("Running on %s\n", mDeviceProperties.name);

		CUDA_CHECK(cuCtxGetCurrent(&mCudaContext));

		OPTIX_CHECK(optixDeviceContextCreate(mCudaContext, 0, &mOptixContext));
		OPTIX_CHECK(optixDeviceContextSetLogCallback(mOptixContext, OptixLogCallback, nullptr, 4));
	}



	void Renderer::CreateDenoiser()
	{
		// create denoiser
		OptixDenoiserOptions denoiserOptions;
		denoiserOptions.inputKind   = OPTIX_DENOISER_INPUT_RGB;
		denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;

		OPTIX_CHECK(optixDenoiserCreate(mOptixContext, &denoiserOptions, &mDenoiser));
		OPTIX_CHECK(optixDenoiserSetModel(mDenoiser, OPTIX_DENOISER_MODEL_KIND_LDR, nullptr, 0));
	}



	void Renderer::CreateModule()
	{
		// module compile options
		mModuleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#if false
		mModuleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
		mModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
		mModuleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
		mModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

		// pipeline compile options
		mPipelineCompileOptions                                  = {};
		mPipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
		mPipelineCompileOptions.usesMotionBlur                   = false;
		mPipelineCompileOptions.numPayloadValues                 = 4;
		mPipelineCompileOptions.numAttributeValues               = 2;
		mPipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
		mPipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

		// pipeline link options
		mPipelineLinkOptions.maxTraceDepth          = 1;
		mPipelineLinkOptions.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
		mPipelineLinkOptions.overrideUsesMotionBlur = 0;

		// load PTX
		const std::string ptxCode = ReadFile("ptx/optix.ptx");
		assert(!ptxCode.empty());

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixModuleCreateFromPTX(mOptixContext, &mModuleCompileOptions, &mPipelineCompileOptions, ptxCode.c_str(), ptxCode.size(), log, &sizeof_log, &mModule));
		if(sizeof_log > 1)
			printf("%s\n", log);
	}



	void Renderer::CreatePrograms()
	{
		auto CreateProgram = [this](const OptixProgramGroupOptions& options, const OptixProgramGroupDesc& desc) -> OptixProgramGroup
		{
			char log[2048];
			size_t logLength = sizeof(log);
			OptixProgramGroup program {};
			OPTIX_CHECK(optixProgramGroupCreate(mOptixContext, &desc, 1, &options, log, &logLength, &program));
			if(logLength > 1)
				printf("%s\n", log);
			return program;
		};

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
	}



	void Renderer::CreatePipeline()
	{
		// create programs vector
		std::vector<OptixProgramGroup> programGroups =
		{
			mRayGenProgram,
			mMissProgram,
			mHitgroupProgram
		};

		char log[2048];
		size_t logLength = sizeof(log);
		OPTIX_CHECK(optixPipelineCreate(mOptixContext, &mPipelineCompileOptions, &mPipelineLinkOptions,
														 programGroups.data(), static_cast<unsigned int>(programGroups.size()),
														 log, &logLength, &mPipeline));
		if(logLength > 1)
			printf("%s\n", log);

		// set stack sizes
		OptixStackSizes stackSizes = {};
		for(auto& g : programGroups)
			OPTIX_CHECK(optixUtilAccumulateStackSizes(g, &stackSizes));

		uint32_t directCallableStackSizeFromTraversal = 2 << 10;
		uint32_t directCallableStackSizeFromState     = 2 << 10;
		uint32_t continuationStackSize                = 2 << 10;
		uint32_t maxTraversableGraphDepth             = 3;
		OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes, mLaunchParams.maxDepth, 0, 0, &directCallableStackSizeFromTraversal,
											   &directCallableStackSizeFromState, &continuationStackSize));
		OPTIX_CHECK(optixPipelineSetStackSize(mPipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState,
											  continuationStackSize, maxTraversableGraphDepth));
	}



	void Renderer::CreateShaderBindingTable()
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



	void Renderer::BuildGeometry(Scene* scene)
	{
		// #TODO: separate instance building from geometry building
		std::vector<OptixBuildInput> buildInputs;
		std::vector<CudaMeshData> meshData;
		std::vector<uint32_t> modelIndices;

		std::vector<OptixInstance> instances;
		uint32_t instanceId = 0;

		if(scene)
		{
			for(auto& inst : scene->Instances())
			{
				const auto& model = inst->GetModel();
				if(model->IsDirty())
				{
					model->Build(mOptixContext, mStream);
					model->BuildLights();
					model->MarkClean();
				}
				instances.push_back(model->InstanceData(instanceId++, inst->Transform()));
				meshData.push_back(model->CudaMesh());

				inst->MarkClean();
			}
		}

		if(meshData.size() == 0)
		{
			mSceneRoot = 0;
			mCudaMeshData.Free();
			mInstancesBuffer.Free();
		}
		else
		{
			// CUDA mesh data
			mCudaMeshData.Upload(meshData, true);

			// upload instances
			mInstancesBuffer.Upload(instances, true);

			// upload lights
			std::vector<LightTriangle> lights = scene->Lights();
			if(lights.size() == 0)
				mCudaLightsBuffer.Free();
			else
				mCudaLightsBuffer.Upload(lights, true);

			SetCudaLights(mCudaLightsBuffer.Ptr<LightTriangle>());
			SetCudaLightCount(static_cast<int32_t>(lights.size()));

			// build top-level
			OptixBuildInput instanceBuildInput = {};
			instanceBuildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
			instanceBuildInput.instanceArray.instances    = mInstancesBuffer.DevicePtr();
			instanceBuildInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());
			buildInputs.push_back(instanceBuildInput);

			// Acceleration setup
			OptixAccelBuildOptions buildOptions = {};
			buildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
			buildOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

			OptixAccelBufferSizes accelBufferSizes = {};
			OPTIX_CHECK(optixAccelComputeMemoryUsage(mOptixContext, &buildOptions, buildInputs.data(), static_cast<unsigned int>(buildInputs.size()), &accelBufferSizes));

			// Execute build
			CudaBuffer tempBuffer(accelBufferSizes.tempSizeInBytes);
			mAccelBuffer.Resize(accelBufferSizes.outputSizeInBytes);
			OPTIX_CHECK(optixAccelBuild(mOptixContext, nullptr, &buildOptions, buildInputs.data(), static_cast<unsigned int>(buildInputs.size()),
										tempBuffer.DevicePtr(), tempBuffer.Size(), mAccelBuffer.DevicePtr(), mAccelBuffer.Size(), &mSceneRoot, nullptr, 0));

			CUDA_CHECK(cudaDeviceSynchronize());
		}

		SetCudaMeshData(mCudaMeshData.Ptr<CudaMeshData>());
	}



	void Renderer::BuildMaterials(Scene* scene)
	{
		// #TODO: only build dirty materials
		const size_t modelCount = scene ? scene->MaterialCount() : 0;
		if(modelCount == 0)
		{
			mCudaMaterialOffsets.Free();
			mCudaMaterialData.Free();
		}
		else
		{
			uint32_t lastMaterialOffset = 0;
			std::vector<CudaMatarial> materialData;
			std::vector<uint32_t> materialOffsets;

			std::vector<uint32_t> modelIndices;
			std::vector<std::shared_ptr<Model>> parsedModels;

			for(auto& inst : scene->Instances())
			{
				const auto& model = inst->GetModel();
				auto it = std::find(parsedModels.begin(), parsedModels.end(), model);
				if(it != parsedModels.end())
				{
					modelIndices.push_back(static_cast<uint32_t>(std::distance(parsedModels.begin(), it)));
				}
				else
				{
					for(auto& mat : model->Materials())
					{
						CudaMatarial m = {};

						m.diffuse = mat->Diffuse();
						m.emissive = mat->Emissive();

						if(mat->DiffuseMap())
						{
							if(mat->DiffuseMap()->IsDirty())
							{
								mat->DiffuseMap()->Build();
								mat->DiffuseMap()->MarkClean();
							}
							m.textures |= Texture_DiffuseMap;
							m.diffuseMap = mat->DiffuseMap()->CudaObject();
						}

						mat->MarkClean();
						materialData.push_back(m);
					}

					materialOffsets.push_back(lastMaterialOffset);
					lastMaterialOffset += static_cast<uint32_t>(model->Materials().size());

					modelIndices.push_back(static_cast<uint32_t>(parsedModels.size()));
					parsedModels.push_back(model);
				}
			}
			mCudaMaterialOffsets.Upload(materialOffsets, true);
			mCudaMaterialData.Upload(materialData, true);
			mCudaModelIndices.Upload(modelIndices, true);
		}

		SetCudaMatarialData(mCudaMaterialData.Ptr<CudaMatarial>());
		SetCudaMatarialOffsets(mCudaMaterialOffsets.Ptr<uint32_t>());
		SetCudaModelIndices(mCudaModelIndices.Ptr<uint32_t>());
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// Renderer::TimeEvent
	//--------------------------------------------------------------------------------------------------------------------------
	Renderer::TimeEvent::TimeEvent()
	{
		CUDA_CHECK(cudaEventCreate(&mStart));
		CUDA_CHECK(cudaEventCreate(&mEnd));

		CUDA_CHECK(cudaEventRecord(mStart));
		CUDA_CHECK(cudaEventRecord(mEnd));
	}



	Renderer::TimeEvent::~TimeEvent()
	{
		CUDA_CHECK(cudaEventDestroy(mStart));
		CUDA_CHECK(cudaEventDestroy(mEnd));
	}



	void Renderer::TimeEvent::Start(cudaStream_t stream)
	{
		CUDA_CHECK(cudaEventRecord(mStart, stream));
	}



	void Renderer::TimeEvent::Stop(cudaStream_t stream)
	{
		CUDA_CHECK(cudaEventRecord(mEnd, stream));
	}



	float Renderer::TimeEvent::Elapsed() const
	{
		float elapsed = 0;
		CUDA_CHECK(cudaEventElapsedTime(&elapsed, mStart, mEnd));
		return elapsed;
	}

}
