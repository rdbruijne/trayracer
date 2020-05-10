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
		enum class OptixRenderModes
		{
			SPT,
			RayPick
		};



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



		std::string ToString(OptixRenderModes renderMode)
		{
			return std::string(magic_enum::enum_name(renderMode));
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
		CreateShaderBindingTables();

		// allocate launch params
		mLaunchParamsBuffer.Alloc(sizeof(LaunchParams));

		// set launch param constants
		mLaunchParams.maxDepth  = 16;
		mLaunchParams.epsilon   = Epsilon;
		mLaunchParams.aoDist    = 1500.f;
		mLaunchParams.zDepthMax = 1500.f;
	}



	Renderer::~Renderer()
	{
		// #TODO: proper cleanup
		if(mCudaGraphicsResource)
			CUDA_CHECK(cudaGraphicsUnregisterResource(mCudaGraphicsResource));
	}



	void Renderer::BuildScene(Scene* scene)
	{
		// #TODO: acync?
		BuildTextures(scene);
		BuildMaterials(scene);
		BuildGeometry(scene);
		mLaunchParams.sampleCount = 0;
	}



	void Renderer::RenderFrame(GLTexture* renderTexture)
	{
		const int2 texRes = renderTexture->Resolution();
		if(texRes.x == 0 || texRes.y == 0)
			return;

		if(mLaunchParams.resX != texRes.x || mLaunchParams.resY != texRes.y)
			Resize(texRes);

		// update scene root
		if(mLaunchParams.sceneRoot != mSceneRoot)
		{
			mLaunchParams.sampleCount = 0;
			mLaunchParams.sceneRoot = mSceneRoot;
		}

		// prepare SPT buffers
		if(mPathStates.Size() == 0)
		{
			mPathStates.Resize(sizeof(float4) * mLaunchParams.resX * mLaunchParams.resY * 3);
			mHitData.Resize(sizeof(uint4) * mLaunchParams.resX * mLaunchParams.resY);

			mLaunchParams.pathStates = mPathStates.Ptr<float4>();
			mLaunchParams.hitData = mHitData.Ptr<uint4>();
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

		// loop
		uint32_t pathCount = mLaunchParams.resX * mLaunchParams.resY;
		const uint32_t stride = mLaunchParams.resX * mLaunchParams.resY;
		for(int pathLength = 0; pathLength < mLaunchParams.maxDepth; pathLength++)
		{
			// launch Optix
			if(pathLength == 0)
			{
				// primary
				InitCudaCounters();
				OPTIX_CHECK(optixLaunch(mPipeline, mStream, mLaunchParamsBuffer.DevicePtr(), mLaunchParamsBuffer.Size(),
										&mRenderModeConfigs[magic_enum::enum_integer(OptixRenderModes::SPT)].shaderBindingTable,
										static_cast<unsigned int>(mLaunchParams.resX), static_cast<unsigned int>(mLaunchParams.resY), 1));
			}
			else if(pathCount > 0)
			{
				// bounce
				mLaunchParams.rayGenMode = RayGen_Secondary;
				mLaunchParamsBuffer.Upload(&mLaunchParams);
				InitCudaCounters();
				OPTIX_CHECK(optixLaunch(mPipeline, mStream, mLaunchParamsBuffer.DevicePtr(), mLaunchParamsBuffer.Size(),
										&mRenderModeConfigs[magic_enum::enum_integer(OptixRenderModes::SPT)].shaderBindingTable,
										pathCount, 1, 1));
			}

			Shade(mRenderMode, pathCount, mColorBuffer.Ptr<float4>(), mPathStates.Ptr<float4>(), mHitData.Ptr<uint4>(),
					make_int2(mLaunchParams.resX, mLaunchParams.resY), stride, pathLength);
			CUDA_CHECK(cudaDeviceSynchronize());

			mCountersBuffer.Download(&counters, 1);
			pathCount = counters.extendRays;
		}

		// run denoiser
		if(ShouldDenoise())
		{
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
			denoiserParams.blendFactor  = 1.f / (mLaunchParams.sampleCount + 1);

			OPTIX_CHECK(optixDenoiserInvoke(mDenoiser, 0, &denoiserParams, mDenoiserState.DevicePtr(), mDenoiserState.Size(),
											&inputLayer, 1, 0, 0, &outputLayer,
											mDenoiserScratch.DevicePtr(), mDenoiserScratch.Size()));

			mDenoisedFrame = true;
		}
		else
		{
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

		// update sample count
		mLaunchParams.sampleCount++;
	}



	void Renderer::DownloadPixels(std::vector<float4>& dstPixels)
	{
		dstPixels.resize(static_cast<size_t>(mLaunchParams.resX) * mLaunchParams.resY);
		if(mDenoisedFrame)
			mDenoisedBuffer.Download(dstPixels);
		else
			mColorBuffer.Download(dstPixels);
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
		mLaunchParams.rayPickPixel = pixelIndex;
		mLaunchParams.rayPickResult = resultBuffer.Ptr<RayPickResult>();

		// upload launch params
		mLaunchParamsBuffer.Upload(&mLaunchParams);

		// launch the kernel
		OPTIX_CHECK(optixLaunch(mPipeline, mStream, mLaunchParamsBuffer.DevicePtr(), mLaunchParamsBuffer.Size(),
								&mRenderModeConfigs[magic_enum::enum_integer(OptixRenderModes::RayPick)].shaderBindingTable, 1, 1, 1));
		CUDA_CHECK(cudaDeviceSynchronize());

		// read the raypick result
		RayPickResult result;
		resultBuffer.Download(&result);

		return result;
	}



	void Renderer::Resize(const int2& resolution)
	{
		// resize buffers
		mColorBuffer.Resize(sizeof(float4) * resolution.x * resolution.y);
		mDenoisedBuffer.Resize(sizeof(float4) * resolution.x * resolution.y);

		// update launch params
		mLaunchParams.sampleCount = 0;
		mLaunchParams.resX = resolution.x;
		mLaunchParams.resY = resolution.y;
		mLaunchParams.accumulator = mColorBuffer.Ptr<float4>();

		// allocate denoiser memory
		OptixDenoiserSizes denoiserReturnSizes;
		OPTIX_CHECK(optixDenoiserComputeMemoryResources(mDenoiser, resolution.x, resolution.y, &denoiserReturnSizes));
		mDenoiserScratch.Resize(denoiserReturnSizes.recommendedScratchSizeInBytes);
		mDenoiserState.Resize(denoiserReturnSizes.stateSizeInBytes);
		OPTIX_CHECK(optixDenoiserSetup(mDenoiser, 0, resolution.x, resolution.y, mDenoiserState.DevicePtr(), mDenoiserState.Size(), mDenoiserScratch.DevicePtr(), mDenoiserScratch.Size()));
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
#ifdef _DEBUG
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
		mPipelineLinkOptions.maxTraceDepth          = 2;
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

		// configs for rendermodes
		mRenderModeConfigs.resize(magic_enum::enum_count<OptixRenderModes>());
		for(size_t m = 0; m < magic_enum::enum_count<OptixRenderModes>(); m++)
		{
			const std::string modeName = ToString(static_cast<OptixRenderModes>(m));

			// entry names
			const std::string raygenEntryName     = "__raygen__" + modeName;
			const std::string missEntryName       = "__miss__" + modeName;
			const std::string anyhitEntryName     = "__anyhit__" + modeName;
			const std::string closesthitEntryName = "__closesthit__" + modeName;

			// ray gen
			OptixProgramGroupDesc raygenDesc      = {};
			raygenDesc.kind                       = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
			raygenDesc.raygen.module              = mModule;
			raygenDesc.raygen.entryFunctionName   = raygenEntryName.c_str();
			mRenderModeConfigs[m].rayGenProgram   = CreateProgram({}, raygenDesc);

			// miss
			OptixProgramGroupDesc missDesc        = {};
			missDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_MISS;
			missDesc.miss.module                  = mModule;
			missDesc.miss.entryFunctionName       = missEntryName.c_str();
			mRenderModeConfigs[m].missProgram     = CreateProgram({}, missDesc);

			// hit
			OptixProgramGroupDesc hitDesc         = {};
			hitDesc.kind                          = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			hitDesc.hitgroup.moduleCH             = mModule;
			hitDesc.hitgroup.entryFunctionNameCH  = closesthitEntryName.c_str();
			hitDesc.hitgroup.moduleAH             = mModule;
			hitDesc.hitgroup.entryFunctionNameAH  = anyhitEntryName.c_str();
			mRenderModeConfigs[m].hitgroupProgram = CreateProgram({}, hitDesc);
		}
	}



	void Renderer::CreatePipeline()
	{
		// count total number of programs
		size_t programCount = mRenderModeConfigs.size() * 3;

		// add programs
		std::vector<OptixProgramGroup> programGroups;
		programGroups.reserve(programCount);
		for(RenderModeConfig& c : mRenderModeConfigs)
		{
			programGroups.push_back(c.rayGenProgram);
			programGroups.push_back(c.missProgram);
			programGroups.push_back(c.hitgroupProgram);
		}

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



	void Renderer::CreateShaderBindingTables()
	{
		for(RenderModeConfig& config : mRenderModeConfigs)
		{
			// raygen records
			RaygenRecord raygen = {};
			OPTIX_CHECK(optixSbtRecordPackHeader(config.rayGenProgram, &raygen));
			config.rayGenRecordsBuffer.Upload(&raygen, 1, true);
			config.shaderBindingTable.raygenRecord = config.rayGenRecordsBuffer.DevicePtr();

			// miss records
			MissRecord miss = {};
			OPTIX_CHECK(optixSbtRecordPackHeader(config.missProgram, &miss));
			config.missRecordsBuffer.Upload(&miss, 1, true);
			config.shaderBindingTable.missRecordBase          = config.missRecordsBuffer.DevicePtr();
			config.shaderBindingTable.missRecordStrideInBytes = sizeof(MissRecord);
			config.shaderBindingTable.missRecordCount         = 1;

			// hitgroup records
			HitgroupRecord hit = {};
			OPTIX_CHECK(optixSbtRecordPackHeader(config.hitgroupProgram, &hit));
			config.hitRecordsBuffer.Upload(&hit, 1, true);
			config.shaderBindingTable.hitgroupRecordBase          = config.hitRecordsBuffer.DevicePtr();
			config.shaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
			config.shaderBindingTable.hitgroupRecordCount         = 1;
		}
	}



	void Renderer::BuildGeometry(Scene* scene)
	{
		std::vector<OptixBuildInput> buildInputs;
		std::vector<CudaMeshData> meshData;

		std::vector<OptixInstance> instances;
		uint32_t instanceId = 0;

		if(scene)
		{
			for(auto& inst : scene->Instances())
			{
				const auto& model = inst->GetModel();
				if(model->IsDirty())
				{
					model->Build(mOptixContext, nullptr);
					model->MarkClean();
				}
				instances.push_back(model->InstanceData(instanceId++, inst->Transform()));
				if(mCudaMeshData.Size() == 0)
					meshData.push_back(model->CudaMesh());

				inst->MarkClean();
			}
		}

		if(mCudaMeshData.Size() == 0 && meshData.size() == 0)
		{
			mSceneRoot = 0;
			mCudaMeshData.Free();
		}
		else
		{
			// CUDA mesh data
			if(mCudaMeshData.Size() == 0)
				mCudaMeshData.Upload(meshData, true);

			// upload instances
			mInstancesBuffer.Upload(instances, true);

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
			static CudaBuffer outputBuffer;
			outputBuffer.Resize(accelBufferSizes.outputSizeInBytes);
			OPTIX_CHECK(optixAccelBuild(mOptixContext, nullptr, &buildOptions, buildInputs.data(), static_cast<unsigned int>(buildInputs.size()),
										tempBuffer.DevicePtr(), tempBuffer.Size(), outputBuffer.DevicePtr(), outputBuffer.Size(), &mSceneRoot, nullptr, 0));

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
			std::vector<CudaMatarial> materialData;
			std::vector<uint32_t> materialOffsets;
			uint32_t lastOffset = 0;
			for(auto& model : scene->Models())
			{
				for(auto& mat : model->Materials())
				{
					CudaMatarial m;

					m.diffuse = mat->Diffuse();
					m.emissive = mat->Emissive();

					if(mat->DiffuseMap())
					{
						m.textures |= Texture_DiffuseMap;
						m.diffuseMap = mTextures[mat->DiffuseMap()]->mObject;
					}

					mat->MarkClean();
					materialData.push_back(m);
				}

				materialOffsets.push_back(lastOffset);
				lastOffset += static_cast<uint32_t>(model->Materials().size());
			}
			mCudaMaterialOffsets.Upload(materialOffsets, true);
			mCudaMaterialData.Upload(materialData, true);
		}
		SetCudaMatarialOffsets(mCudaMaterialOffsets.Ptr<uint32_t>());
		SetCudaMatarialData(mCudaMaterialData.Ptr<CudaMatarial>());
	}



	void Renderer::BuildTextures(Scene* scene)
	{
		if(!scene)
		{
			mTextures.clear();
			return;
		}

		// gather textures
		std::set<std::shared_ptr<Texture>> textures;
		for(auto& model : scene->Models())
		{
			for(auto& mat : model->Materials())
			{
				if(mat->DiffuseMap())
					textures.insert(mat->DiffuseMap());
			}
		}

		// remove removed textures
		std::vector<std::shared_ptr<Texture>> texturesToRemove;
		for(auto& t : mTextures)
		{
			if(textures.find(t.first) == textures.end())
				texturesToRemove.push_back(t.first);
		}
		for(auto& t : texturesToRemove)
			mTextures.erase(t);

		// add new textures
		for(auto& t : textures)
		{
			if(t->IsDirty() || mTextures.find(t) == mTextures.end())
				mTextures[t] = std::make_shared<CudaTexture>(t);
			t->MarkClean();
		}
	}



	Renderer::CudaTexture::CudaTexture(std::shared_ptr<Texture> srcTex)
	{
		// create channel descriptor
		constexpr uint32_t numComponents = 4;
		const uint32_t width  = srcTex->Resolution().x;
		const uint32_t height = srcTex->Resolution().y;
		const uint32_t pitch  = width * numComponents * sizeof(uint8_t);
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

		// upload pixels
		CUDA_CHECK(cudaMallocArray(&mArray, &channelDesc, width, height));
		CUDA_CHECK(cudaMemcpy2DToArray(mArray, 0, 0, srcTex->Pixels().data(), pitch, pitch, height, cudaMemcpyHostToDevice));

		// resource descriptor
		cudaResourceDesc resourceDesc = {};
		resourceDesc.resType = cudaResourceTypeArray;
		resourceDesc.res.array.array = mArray;

		// texture descriptor
		cudaTextureDesc texDesc     = {};
		texDesc.addressMode[0]      = cudaAddressModeWrap;
		texDesc.addressMode[1]      = cudaAddressModeWrap;
		texDesc.filterMode          = cudaFilterModeLinear;
		texDesc.readMode            = cudaReadModeNormalizedFloat;
		texDesc.sRGB                = 0;
		texDesc.borderColor[0]      = 1.0f;
		texDesc.normalizedCoords    = 1;
		texDesc.maxAnisotropy       = 1;
		texDesc.mipmapFilterMode    = cudaFilterModePoint;
		texDesc.minMipmapLevelClamp = 0;
		texDesc.maxMipmapLevelClamp = 99;

		// texture object
		CUDA_CHECK(cudaCreateTextureObject(&mObject, &resourceDesc, &texDesc, nullptr));
	}



	Renderer::CudaTexture::~CudaTexture()
	{
		if(mArray)
			CUDA_CHECK(cudaFreeArray(mArray));
		if(mObject)
			CUDA_CHECK(cudaDestroyTextureObject(mObject));
	}
}
