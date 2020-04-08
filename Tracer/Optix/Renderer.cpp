#include "Optix/Renderer.h"

// Project
#include "OpenGL/GLTexture.h"
#include "Optix/OptixHelpers.h"
#include "Resources/Scene.h"
#include "Resources/Material.h"
#include "Resources/Mesh.h"
#include "Resources/Model.h"
#include "Resources/Texture.h"
#include "Utility/LinearMath.h"
#include "Utility/Utility.h"

// CUDA
#include <cuda_gl_interop.h>

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



		std::string EntryName(Renderer::RenderModes renderMode, const std::string& entryPoint)
		{
			return entryPoint + ToString(renderMode);
		}
	}



	Renderer::Renderer()
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
		BuildShaderBindingTables(nullptr);

		// allocate launch params
		mLaunchParamsBuffer.Alloc(sizeof(LaunchParams));

		// set launch param constants
		mLaunchParams.maxDepth = 2;
		mLaunchParams.epsilon = Epsilon;
		mLaunchParams.aoDist = 10.f;
		mLaunchParams.zDepthMaX = 10.f;
	}



	Renderer::~Renderer()
	{
		// #TODO(RJCDB): cleanup
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

		if(mLaunchParams.resolutionX != texRes.x || mLaunchParams.resolutionY != texRes.y)
		{
			// resize buffer
			mColorBuffer.Resize(sizeof(float4) * texRes.x * texRes.y);

			// update launch params
			mLaunchParams.sampleCount = 0;
			mLaunchParams.resolutionX = texRes.x;
			mLaunchParams.resolutionY = texRes.y;
			mLaunchParams.colorBuffer = reinterpret_cast<float4*>(mColorBuffer.DevicePtr());
		}

		// update scene root
		if(mLaunchParams.sceneRoot != mSceneRoot)
		{
			mLaunchParams.sampleCount = 0;
			mLaunchParams.sceneRoot = mSceneRoot;
		}

		// upload launch params
		mLaunchParamsBuffer.Upload(&mLaunchParams, 1);

		// launch OptiX
		OPTIX_CHECK(optixLaunch(mPipeline, mStream, mLaunchParamsBuffer.DevicePtr(), mLaunchParamsBuffer.Size(),
								&mRenderModeConfigs[magic_enum::enum_integer(mRenderMode)].shaderBindingTable,
								static_cast<unsigned int>(mLaunchParams.resolutionX), static_cast<unsigned int>(mLaunchParams.resolutionY), 1));
		CUDA_CHECK(cudaDeviceSynchronize());

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
		CUDA_CHECK(cudaMemcpy2DToArray(cudaTexPtr, 0, 0, mColorBuffer.Ptr(), texRes.x * sizeof(float4),texRes.x * sizeof(float4), texRes.y, cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaGraphicsUnmapResources(1, &mCudaGraphicsResource, 0));
		CUDA_CHECK(cudaDeviceSynchronize());

		// update sample count
		mLaunchParams.sampleCount++;
	}



	void Renderer::DownloadPixels(std::vector<float4>& dstPixels)
	{
		dstPixels.resize(static_cast<size_t>(mLaunchParams.resolutionX) * mLaunchParams.resolutionY);
		mColorBuffer.Download(dstPixels.data(), static_cast<size_t>(mLaunchParams.resolutionX) * mLaunchParams.resolutionY);
	}



	void Renderer::SetCamera(float3 cameraPos, float3 cameraForward, float3 cameraUp, float camFov)
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



	void Renderer::CreateContext()
	{
		const int deviceID = 0;
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



	void Renderer::CreateRaygenPrograms()
	{
		for(size_t m = 0; m < magic_enum::enum_count<RenderModes>(); m++)
		{
			mRenderModeConfigs[m].rayGenPrograms.resize(1);

			const std::string entryName = EntryName(static_cast<RenderModes>(m), "__raygen__");

			OptixProgramGroupOptions options = {};
			OptixProgramGroupDesc desc       = {};
			desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
			desc.raygen.module               = mModule;
			desc.raygen.entryFunctionName    = entryName.c_str();

			char log[2048];
			size_t sizeof_log = sizeof(log);
			OPTIX_CHECK(optixProgramGroupCreate(mOptixContext, &desc, 1, &options, log, &sizeof_log, &mRenderModeConfigs[m].rayGenPrograms[0]));
			if(sizeof_log > 1)
				printf("%s\n", log);
		}
	}



	void Renderer::CreateMissPrograms()
	{
		for(size_t m = 0; m < magic_enum::enum_count<RenderModes>(); m++)
		{
			mRenderModeConfigs[m].missPrograms.resize(1);

			const std::string entryName = EntryName(static_cast<RenderModes>(m), "__miss__");

			OptixProgramGroupOptions options = {};
			OptixProgramGroupDesc desc       = {};
			desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_MISS;
			desc.miss.module                 = mModule;
			desc.miss.entryFunctionName      = entryName.c_str();

			char log[2048];
			size_t logLength = sizeof(log);
			OPTIX_CHECK(optixProgramGroupCreate(mOptixContext, &desc, 1, &options, log, &logLength, &mRenderModeConfigs[m].missPrograms[0]));
			if(logLength > 1)
				printf("%s\n", log);
		}
	}



	void Renderer::CreateHitgroupPrograms()
	{
		for(size_t m = 0; m < magic_enum::enum_count<RenderModes>(); m++)
		{
			mRenderModeConfigs[m].hitgroupPrograms.resize(1);

			const std::string ahEntryName = EntryName(static_cast<RenderModes>(m), "__anyhit__");
			const std::string chEntryName = EntryName(static_cast<RenderModes>(m), "__closesthit__");

			OptixProgramGroupOptions options  = {};
			OptixProgramGroupDesc desc        = {};
			desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			desc.hitgroup.moduleCH            = mModule;
			desc.hitgroup.entryFunctionNameCH = chEntryName.c_str();
			desc.hitgroup.moduleAH            = mModule;
			desc.hitgroup.entryFunctionNameAH = ahEntryName.c_str();

			mRenderModeConfigs[m].hitgroupPrograms.resize(1);

			char log[2048];
			size_t logLength = sizeof(log);
			OPTIX_CHECK(optixProgramGroupCreate(mOptixContext, &desc, 1, &options, log, &logLength, &mRenderModeConfigs[m].hitgroupPrograms[0]));
			if(logLength > 1)
				printf("%s\n", log);
		}
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
		const uint32_t directCallableStackSizeFromTraversal = 2 << 10;
		const uint32_t directCallableStackSizeFromState     = 2 << 10;
		const uint32_t continuationStackSize                = 2 << 10;
		const uint32_t maxTraversableGraphDepth             = 3;
		OPTIX_CHECK(optixPipelineSetStackSize(mPipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState, continuationStackSize, maxTraversableGraphDepth));
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
					mVertexBuffers[meshIx].AllocAndUpload(mesh->Vertices());
					mNormalBuffers[meshIx].AllocAndUpload(mesh->Normals());
					mTexcoordBuffers[meshIx].AllocAndUpload(mesh->Texcoords());
					mIndexBuffers[meshIx].AllocAndUpload(mesh->Indices());

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
					const uint32_t buildFlags[] = { 0 };
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
			compactedSizeBuffer.Download(&compactedSize, 1);

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

			config.rayGenRecordsBuffer.Free();
			config.rayGenRecordsBuffer.AllocAndUpload(raygenRecords);
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

			config.missRecordsBuffer.Free();
			config.missRecordsBuffer.AllocAndUpload(missRecords);
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

			config.hitRecordsBuffer.Free();
			config.hitRecordsBuffer.AllocAndUpload(hitgroupRecords);
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
		const uint32_t width = srcTex->mResolution.x;
		const uint32_t height = srcTex->mResolution.y;
		const uint32_t numComponents = 4;
		const uint32_t pitch = width * numComponents * sizeof(uint8_t);
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
