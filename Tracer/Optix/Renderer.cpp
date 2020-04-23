#include "Optix/Renderer.h"

// Project
#include "OpenGL/GLTexture.h"
#include "Resources/CameraNode.h"
#include "Resources/Scene.h"
#include "Resources/Material.h"
#include "Resources/Mesh.h"
#include "Resources/Model.h"
#include "Resources/Texture.h"
#include "Utility/LinearMath.h"
#include "Utility/Utility.h"

// OptiX
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
#include <string>



namespace Tracer
{
	namespace
	{
		constexpr size_t RayPickConfigIx = magic_enum::enum_count<Renderer::RenderModes>();



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



		std::string EntryName(Renderer::RenderModes renderMode, const std::string& entryPoint)
		{
			return entryPoint + ToString(renderMode);
		}



		std::string ToString(OptixResult optixResult)
		{
			switch(optixResult)
			{
			case OPTIX_SUCCESS:
				return "OPTIX_SUCCESS";

			case OPTIX_ERROR_INVALID_VALUE:
				return "OPTIX_ERROR_INVALID_VALUE";

			case OPTIX_ERROR_HOST_OUT_OF_MEMORY:
				return "OPTIX_ERROR_HOST_OUT_OF_MEMORY";

			case OPTIX_ERROR_INVALID_OPERATION:
				return "OPTIX_ERROR_INVALID_OPERATION";

			case OPTIX_ERROR_FILE_IO_ERROR:
				return "OPTIX_ERROR_FILE_IO_ERROR";

			case OPTIX_ERROR_INVALID_FILE_FORMAT:
				return "OPTIX_ERROR_INVALID_FILE_FORMAT";

			case OPTIX_ERROR_DISK_CACHE_INVALID_PATH:
				return "OPTIX_ERROR_DISK_CACHE_INVALID_PATH";

			case OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR:
				return "OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR";

			case OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR:
				return "OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR";

			case OPTIX_ERROR_DISK_CACHE_INVALID_DATA:
				return "OPTIX_ERROR_DISK_CACHE_INVALID_DATA";

			case OPTIX_ERROR_LAUNCH_FAILURE:
				return "OPTIX_ERROR_LAUNCH_FAILURE";

			case OPTIX_ERROR_INVALID_DEVICE_CONTEXT:
				return "OPTIX_ERROR_INVALID_DEVICE_CONTEXT";

			case OPTIX_ERROR_CUDA_NOT_INITIALIZED:
				return "OPTIX_ERROR_CUDA_NOT_INITIALIZED";

			case OPTIX_ERROR_INVALID_PTX:
				return "OPTIX_ERROR_INVALID_PTX";

			case OPTIX_ERROR_INVALID_LAUNCH_PARAMETER:
				return "OPTIX_ERROR_INVALID_LAUNCH_PARAMETER";

			case OPTIX_ERROR_INVALID_PAYLOAD_ACCESS:
				return "OPTIX_ERROR_INVALID_PAYLOAD_ACCESS";

			case OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS:
				return "OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS";

			case OPTIX_ERROR_INVALID_FUNCTION_USE:
				return "OPTIX_ERROR_INVALID_FUNCTION_USE";

			case OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS:
				return "OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS";

			case OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY:
				return "OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY";

			case OPTIX_ERROR_PIPELINE_LINK_ERROR:
				return "OPTIX_ERROR_PIPELINE_LINK_ERROR";

			case OPTIX_ERROR_INTERNAL_COMPILER_ERROR:
				return "OPTIX_ERROR_INTERNAL_COMPILER_ERROR";

			case OPTIX_ERROR_DENOISER_MODEL_NOT_SET:
				return "OPTIX_ERROR_DENOISER_MODEL_NOT_SET";

			case OPTIX_ERROR_DENOISER_NOT_INITIALIZED:
				return "OPTIX_ERROR_DENOISER_NOT_INITIALIZED";

			case OPTIX_ERROR_ACCEL_NOT_COMPATIBLE:
				return "OPTIX_ERROR_ACCEL_NOT_COMPATIBLE";

			case OPTIX_ERROR_NOT_SUPPORTED:
				return "OPTIX_ERROR_NOT_SUPPORTED";

			case OPTIX_ERROR_UNSUPPORTED_ABI_VERSION:
				return "OPTIX_ERROR_UNSUPPORTED_ABI_VERSION";

			case OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH:
				return "OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH";

			case OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS:
				return "OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS";

			case OPTIX_ERROR_LIBRARY_NOT_FOUND:
				return "OPTIX_ERROR_LIBRARY_NOT_FOUND";

			case OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND:
				return "OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND";

			case OPTIX_ERROR_CUDA_ERROR:
				return "OPTIX_ERROR_CUDA_ERROR";

			case OPTIX_ERROR_INTERNAL_ERROR:
				return "OPTIX_ERROR_INTERNAL_ERROR";

			case OPTIX_ERROR_UNKNOWN:
				return "OPTIX_ERROR_UNKNOWN";

			default:
				return "Unknown OptiX error code!";
			}
		}



#define OPTIX_CHECK(x) OptixCheck((x), __FILE__, __LINE__)
		void OptixCheck(OptixResult res, const char* file, int line)
		{
			//assert(res == OPTIX_SUCCESS);
			if(res != OPTIX_SUCCESS)
			{
				const std::string errorMessage = format("OptiX error at \"%s\" @ %i: %s", file, line, ToString(res).c_str());
				throw std::runtime_error(errorMessage);
			}
		}
	}



	Renderer::Renderer()
	{
		// create OptiX content
		CreateContext();
		CreateModule();
		CreatePrograms();
		CreatePipeline();
		CreateDenoiser();

		// build shader binding table
		BuildShaderBindingTables(nullptr);

		// allocate launch params
		mLaunchParamsBuffer.Alloc(sizeof(LaunchParams));

		// set launch param constants
		mLaunchParams.maxDepth  = 2;
		mLaunchParams.epsilon   = Epsilon;
		mLaunchParams.aoDist    = 10.f;
		mLaunchParams.zDepthMax = 10.f;
	}



	Renderer::~Renderer()
	{
		// #TODO(RJCDB): proper cleanup
		if(mCudaGraphicsResource)
			CUDA_CHECK(cudaGraphicsUnregisterResource(mCudaGraphicsResource));
	}



	void Renderer::BuildScene(Scene* scene)
	{
		BuildGeometry(scene);
		BuildTextures(scene);
		BuildShaderBindingTables(scene);
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

		// upload launch params
		mLaunchParamsBuffer.Upload(mLaunchParams);

		// launch OptiX
		OPTIX_CHECK(optixLaunch(mPipeline, mStream, mLaunchParamsBuffer.DevicePtr(), mLaunchParamsBuffer.Size(),
								&mRenderModeConfigs[magic_enum::enum_integer(mRenderMode)].shaderBindingTable,
								static_cast<unsigned int>(mLaunchParams.resX), static_cast<unsigned int>(mLaunchParams.resY), 1));
		CUDA_CHECK(cudaDeviceSynchronize());

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
		if(camNode.HasChanged())
		{
			SetCamera(camNode.Position(), normalize(camNode.Target() - camNode.Position()), camNode.Up(), camNode.Fov());
			camNode.ClearHasChanged();
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
		mLaunchParams.rayPickResult     = reinterpret_cast<RayPickResult*>(resultBuffer.DevicePtr());

		// upload launch params
		mLaunchParamsBuffer.Upload(mLaunchParams);

		// launch the kernel
		OPTIX_CHECK(optixLaunch(mPipeline, mStream, mLaunchParamsBuffer.DevicePtr(), mLaunchParamsBuffer.Size(),
								&mRenderModeConfigs[RayPickConfigIx].shaderBindingTable, 1, 1, 1));
		CUDA_CHECK(cudaDeviceSynchronize());

		// read the raypick result
		RayPickResult result;
		resultBuffer.Download(result);

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
		mLaunchParams.colorBuffer = reinterpret_cast<float4*>(mColorBuffer.DevicePtr());

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
		mPipelineCompileOptions.numPayloadValues                 = 2;
		mPipelineCompileOptions.numAttributeValues               = 2;
		mPipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
		mPipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

		// pipeline link options
		mPipelineLinkOptions.maxTraceDepth          = 2;
		mPipelineLinkOptions.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
		mPipelineLinkOptions.overrideUsesMotionBlur = 0;

		// load PTX
		const std::string ptxCode = ReadFile("ptx/program.ptx");
		assert(!ptxCode.empty());

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixModuleCreateFromPTX(mOptixContext, &mModuleCompileOptions, &mPipelineCompileOptions, ptxCode.c_str(), ptxCode.size(), log, &sizeof_log, &mModule));
		if(sizeof_log > 1)
			printf("%s\n", log);
	}



	void Renderer::CreatePrograms()
	{
		// raygen
		auto CreateRaygenProgram = [this](const std::string& name) -> std::vector<OptixProgramGroup>
		{
			const std::string entryName = "__raygen__" + name;

			OptixProgramGroupOptions options = {};
			OptixProgramGroupDesc desc       = {};
			desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
			desc.raygen.module               = mModule;
			desc.raygen.entryFunctionName    = entryName.c_str();

			char log[2048];
			size_t logLength = sizeof(log);
			std::vector<OptixProgramGroup> programs(1);
			OPTIX_CHECK(optixProgramGroupCreate(mOptixContext, &desc, static_cast<unsigned int>(programs.size()), &options, log, &logLength, programs.data()));
			if(logLength > 1)
				printf("%s\n", log);

			return programs;
		};

		// miss
		auto CreateMissProgram = [this](const std::string& name) -> std::vector<OptixProgramGroup>
		{
			const std::string entryName = "__miss__" + name;

			OptixProgramGroupOptions options = {};
			OptixProgramGroupDesc desc       = {};
			desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_MISS;
			desc.miss.module                 = mModule;
			desc.miss.entryFunctionName      = entryName.c_str();

			char log[2048];
			size_t logLength = sizeof(log);
			std::vector<OptixProgramGroup> programs(1);
			OPTIX_CHECK(optixProgramGroupCreate(mOptixContext, &desc, static_cast<unsigned int>(programs.size()), &options, log, &logLength, programs.data()));
			if(logLength > 1)
				printf("%s\n", log);

			return programs;
		};

		// hit
		auto CreateHitProgram = [this](const std::string& name) -> std::vector<OptixProgramGroup>
		{
			const std::string ahEntryName = "__anyhit__" + name;
			const std::string chEntryName = "__closesthit__" + name;

			OptixProgramGroupOptions options  = {};
			OptixProgramGroupDesc desc        = {};
			desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			desc.hitgroup.moduleCH            = mModule;
			desc.hitgroup.entryFunctionNameCH = chEntryName.c_str();
			desc.hitgroup.moduleAH            = mModule;
			desc.hitgroup.entryFunctionNameAH = ahEntryName.c_str();

			char log[2048];
			size_t logLength = sizeof(log);
			std::vector<OptixProgramGroup> programs(1);
			OPTIX_CHECK(optixProgramGroupCreate(mOptixContext, &desc, static_cast<unsigned int>(programs.size()), &options, log, &logLength, programs.data()));
			if(logLength > 1)
				printf("%s\n", log);

			return programs;
		};

		// configs for rendermodes
		for(size_t m = 0; m < magic_enum::enum_count<RenderModes>(); m++)
		{
			const std::string modeName = ToString(static_cast<RenderModes>(m));
			mRenderModeConfigs[m].rayGenPrograms   = CreateRaygenProgram(modeName);
			mRenderModeConfigs[m].missPrograms     = CreateMissProgram(modeName);
			mRenderModeConfigs[m].hitgroupPrograms = CreateHitProgram(modeName);
		}

		// ray pick
		const std::string rayPickName = "RayPick";
		mRenderModeConfigs[RayPickConfigIx].rayGenPrograms   = CreateRaygenProgram(rayPickName);
		mRenderModeConfigs[RayPickConfigIx].missPrograms     = CreateMissProgram(rayPickName);
		mRenderModeConfigs[RayPickConfigIx].hitgroupPrograms = CreateHitProgram(rayPickName);
	}



	void Renderer::CreatePipeline()
	{
		// count total number of programs
		size_t programCount = 0;
		for(RenderModeConfig& c : mRenderModeConfigs)
			programCount += c.rayGenPrograms.size() + c.missPrograms.size() + c.hitgroupPrograms.size();

		// add programs
		std::vector<OptixProgramGroup> programGroups;
		programGroups.reserve(programCount);
		for(RenderModeConfig& c : mRenderModeConfigs)
		{
			for(OptixProgramGroup& p : c.rayGenPrograms)
				programGroups.push_back(p);
			for(OptixProgramGroup& p : c.missPrograms)
				programGroups.push_back(p);
			for(OptixProgramGroup& p : c.hitgroupPrograms)
				programGroups.push_back(p);
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



	void Renderer::CreateDenoiser()
	{
		// create denoiser
		OptixDenoiserOptions denoiserOptions;
		denoiserOptions.inputKind   = OPTIX_DENOISER_INPUT_RGB;
		denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;

		OPTIX_CHECK(optixDenoiserCreate(mOptixContext, &denoiserOptions, &mDenoiser));
		OPTIX_CHECK(optixDenoiserSetModel(mDenoiser, OPTIX_DENOISER_MODEL_KIND_LDR, nullptr, 0));
	}



	void Renderer::BuildGeometry(Scene* scene)
	{
		std::vector<OptixBuildInput> buildInputs;
		if(scene)
		{
			const size_t meshCount = scene->MeshCount();

			//--------------------------------
			// Build input
			//--------------------------------
			buildInputs.resize(meshCount);
			mVertexBuffers.resize(meshCount);
			mNormalBuffers.resize(meshCount);
			mTexcoordBuffers.resize(meshCount);
			mIndexBuffers.resize(meshCount);

			size_t meshIx = 0;
			for(auto& model : scene->Models())
			{
				for(auto& mesh : model->Meshes())
				{
					OptixBuildInput& bi = buildInputs[meshIx];

					bi = {};
					bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

					// upload buffers
					mVertexBuffers[meshIx].Upload(mesh->Vertices(), true);
					mNormalBuffers[meshIx].Upload(mesh->Normals(), true);
					mTexcoordBuffers[meshIx].Upload(mesh->Texcoords(), true);
					mIndexBuffers[meshIx].Upload(mesh->Indices(), true);

					// vertices
					bi.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
					bi.triangleArray.vertexStrideInBytes = sizeof(float3);
					bi.triangleArray.numVertices         = static_cast<unsigned int>(mesh->Vertices().size());
					bi.triangleArray.vertexBuffers       = mVertexBuffers[meshIx].DevicePtrPtr();

					// indices
					bi.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
					bi.triangleArray.indexStrideInBytes = sizeof(uint3);
					bi.triangleArray.numIndexTriplets   = static_cast<unsigned int>(mesh->Indices().size());
					bi.triangleArray.indexBuffer        = mIndexBuffers[meshIx].DevicePtr();

					// other
					constexpr uint32_t buildFlags[] = { 0 };
					bi.triangleArray.flags                       = buildFlags;
					bi.triangleArray.numSbtRecords               = 1;
					bi.triangleArray.sbtIndexOffsetBuffer        = 0;
					bi.triangleArray.sbtIndexOffsetSizeInBytes   = 0;
					bi.triangleArray.sbtIndexOffsetStrideInBytes = 0;

					meshIx++;
				}
			}
		}

		if(buildInputs.size() == 0)
		{
			mSceneRoot = 0;
			mAccelBuffer.Free();
		}
		else
		{
			//--------------------------------
			// Acceleration setup
			//--------------------------------
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags            = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
			accelOptions.motionOptions.numKeys = 1;
			accelOptions.operation             = OPTIX_BUILD_OPERATION_BUILD;

			OptixAccelBufferSizes accelBufferSizes = {};
			OPTIX_CHECK(optixAccelComputeMemoryUsage(mOptixContext, &accelOptions, buildInputs.data(), static_cast<unsigned int>(buildInputs.size()), &accelBufferSizes));

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
			OPTIX_CHECK(optixAccelBuild(mOptixContext, nullptr, &accelOptions, buildInputs.data(), static_cast<unsigned int>(buildInputs.size()),
										tempBuffer.DevicePtr(), tempBuffer.Size(), outputBuffer.DevicePtr(), outputBuffer.Size(), &mSceneRoot, &emitDesc, 1));
			CUDA_CHECK(cudaDeviceSynchronize());

			//--------------------------------
			// Compact
			//--------------------------------
			uint64_t compactedSize = 0;
			compactedSizeBuffer.Download(compactedSize);

			mAccelBuffer.Alloc(compactedSize);
			OPTIX_CHECK(optixAccelCompact(mOptixContext, nullptr, mSceneRoot, mAccelBuffer.DevicePtr(), mAccelBuffer.Size(), &mSceneRoot));
			CUDA_CHECK(cudaDeviceSynchronize());
		}
	}



	void Renderer::BuildShaderBindingTables(Scene* scene)
	{
		for(RenderModeConfig& config : mRenderModeConfigs)
		{
			// raygen records
			std::vector<RaygenRecord> raygenRecords;
			raygenRecords.reserve(config.rayGenPrograms.size());
			for(auto p : config.rayGenPrograms)
			{
				RaygenRecord r;
				OPTIX_CHECK(optixSbtRecordPackHeader(p, &r));
				r.data = nullptr;
				raygenRecords.push_back(r);
			}

			config.rayGenRecordsBuffer.Upload(raygenRecords, true);
			config.shaderBindingTable.raygenRecord = config.rayGenRecordsBuffer.DevicePtr();

			// miss records
			std::vector<MissRecord> missRecords;
			missRecords.reserve(config.missPrograms.size());
			for(auto p : config.missPrograms)
			{
				MissRecord r;
				OPTIX_CHECK(optixSbtRecordPackHeader(p, &r));
				r.data = nullptr;
				missRecords.push_back(r);
			}

			config.missRecordsBuffer.Upload(missRecords, true);
			config.shaderBindingTable.missRecordBase          = config.missRecordsBuffer.DevicePtr();
			config.shaderBindingTable.missRecordStrideInBytes = sizeof(MissRecord);
			config.shaderBindingTable.missRecordCount         = static_cast<unsigned int>(missRecords.size());

			// hitgroup records
			std::vector<HitgroupRecord> hitgroupRecords;
			const size_t meshCount = scene ? scene->MeshCount() : 0;
			if(meshCount == 0)
			{
				// dummy material
				hitgroupRecords.reserve(1);
				uint64_t objectType = 0;
				HitgroupRecord r;
				OPTIX_CHECK(optixSbtRecordPackHeader(config.hitgroupPrograms[objectType], &r));
				r.meshData.diffuse = make_float3(.75f, 0, .75f);
				hitgroupRecords.push_back(r);
			}
			else
			{
				hitgroupRecords.reserve(meshCount);
				size_t meshIx = 0;
				for(auto& model : scene->Models())
				{
					for(auto& mesh : model->Meshes())
					{
						uint64_t objectType = 0;
						HitgroupRecord r;
						OPTIX_CHECK(optixSbtRecordPackHeader(config.hitgroupPrograms[objectType], &r));

						// buffers
						r.meshData.vertices  = reinterpret_cast<float3*>(mVertexBuffers[meshIx].DevicePtr());
						r.meshData.normals   = reinterpret_cast<float3*>(mNormalBuffers[meshIx].DevicePtr());
						r.meshData.texcoords = reinterpret_cast<float3*>(mTexcoordBuffers[meshIx].DevicePtr());
						r.meshData.indices   = reinterpret_cast<uint3*>(mIndexBuffers[meshIx].DevicePtr());

						// general info
						r.meshData.objectID = static_cast<uint32_t>(meshIx);

						// material data
						auto mat = mesh->Mat();
						r.meshData.diffuse = mat->mDiffuse;
						r.meshData.emissive = mat->mEmissive;

						if(mat->mDiffuseMap)
						{
							r.meshData.textures |= Texture_DiffuseMap;
							r.meshData.diffuseMap = mTextures[mat->mDiffuseMap].mObject;
						}


						hitgroupRecords.push_back(r);
						meshIx++;
					}
				}
			}

			config.hitRecordsBuffer.Upload(hitgroupRecords, true);
			config.shaderBindingTable.hitgroupRecordBase          = config.hitRecordsBuffer.DevicePtr();
			config.shaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
			config.shaderBindingTable.hitgroupRecordCount         = static_cast<unsigned int>(hitgroupRecords.size());
		}
	}



	void Renderer::BuildTextures(Scene* scene)
	{
		mTextures.clear();

		if(!scene)
			return;

		for(auto& model : scene->Models())
		{
			for(auto& mesh : model->Meshes())
			{
				auto mat = mesh->Mat();
				if(mat->mDiffuseMap)
					mTextures[mat->mDiffuseMap] = OptixTexture(mat->mDiffuseMap);
			}
		}
	}



	Renderer::OptixTexture::OptixTexture(std::shared_ptr<Texture> srcTex)
	{
		// create channel descriptor
		constexpr uint32_t numComponents = 4;
		const uint32_t width  = srcTex->mResolution.x;
		const uint32_t height = srcTex->mResolution.y;
		const uint32_t pitch  = width * numComponents * sizeof(uint8_t);
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

		// upload pixels
		CUDA_CHECK(cudaMallocArray(&mArray, &channelDesc, width, height));
		CUDA_CHECK(cudaMemcpy2DToArray(mArray, 0, 0, srcTex->mPixels.data(), pitch, pitch, height, cudaMemcpyHostToDevice));

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



	std::string ToString(Renderer::RenderModes renderMode)
	{
		return std::string(magic_enum::enum_name(renderMode));
	}
}
