#pragma once

// Project
#include "Renderer/CudaBuffer.h"
#include "Common/CommonStructs.h"

// Magic Enum
#pragma warning(push)
#pragma warning(disable: 5027)
#include "magic_enum/magic_enum.hpp"
#pragma warning(pop)

// C++
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace Tracer
{
	class CameraNode;
	class GLTexture;
	class Scene;
	class Texture;
	class Renderer
	{
	public:
		explicit Renderer();
		~Renderer();

		void BuildScene(Scene* scene);
		void RenderFrame(GLTexture* renderTexture);

		void DownloadPixels(std::vector<float4>& dstPixels);

		void SetCamera(CameraNode& camNode);
		void SetCamera(const float3& cameraPos, const float3& cameraForward, const float3& cameraUp, float camFov);

		inline RenderModes RenderMode() const { return mRenderMode; }
		void SetRenderMode(RenderModes mode);

		inline int SampleCount() const { return mLaunchParams.sampleCount; }

		struct RenderStats
		{
			uint64_t pathCount = 0;

			uint64_t primaryPathCount = 0;
			uint64_t secondaryPathCount = 0;
			uint64_t deepPathCount = 0;

			float renderTimeMs = 0;
			float primaryPathTimeMs = 0;
			float secondaryPathTimeMs = 0;
			float deepPathTimeMs = 0;
			float shadeTimeMs = 0;
			float denoiseTimeMs = 0;
		};
		RenderStats Statistics() const { return mRenderStats; }

		// ray picking
		RayPickResult PickRay(int2 pixelIndex);

		// denoising
		inline bool DenoisingEnabled() const { return mDenoisingEnabled; }
		inline void SetDenoiserEnabled(bool enabled) { mDenoisingEnabled = enabled; }

		inline int32_t DenoiserSampleTreshold() const { return mDenoiserSampleTreshold; }
		inline void SetDenoiserSampleTreshold(uint32_t treshold) { mDenoiserSampleTreshold = treshold; }

		// kernel settings
#define KERNEL_SETTING(type, name, func, reqClear)				\
		inline type func() const { return mLaunchParams.name; }	\
		inline void Set##func(type name)						\
		{														\
			if(name != mLaunchParams.name)						\
			{													\
				mLaunchParams.name = name;						\
				if constexpr (reqClear)							\
					mLaunchParams.sampleCount = 0;				\
			}													\
		}

		KERNEL_SETTING(int, multiSample, MultiSample, false)
		KERNEL_SETTING(int, maxDepth, MaxDepth, true)
		KERNEL_SETTING(float, aoDist, AODist, true)
		KERNEL_SETTING(float, zDepthMax, ZDepthMax, true)

#undef KERNEL_SETTING

	private:
		void Resize(const int2& resolution);
		bool ShouldDenoise() const;

		// Creation
		void CreateContext();
		void CreateDenoiser();
		void CreateModule();
		void CreatePrograms();
		void CreatePipeline();
		void CreateShaderBindingTables();

		// scene building
		void BuildGeometry(Scene* scene);
		void BuildMaterials(Scene* scene);
		void BuildTextures(Scene* scene);

		// render mode
		RenderModes mRenderMode = RenderModes::DiffuseFilter;

		// stats
		RenderStats mRenderStats = {};
		cudaEvent_t mRenderStart = nullptr;
		cudaEvent_t mRenderEnd = nullptr;
		cudaEvent_t mTraceStart = nullptr;
		cudaEvent_t mTraceEnd = nullptr;
		cudaEvent_t mShadeStart = nullptr;
		cudaEvent_t mShadeEnd = nullptr;
		cudaEvent_t mDenoiseStart = nullptr;
		cudaEvent_t mDenoiseEnd = nullptr;

		// Render buffer
		CudaBuffer mColorBuffer = {};
		GLTexture* mRenderTarget = nullptr;
		cudaGraphicsResource* mCudaGraphicsResource = nullptr;

		// SPT
		CudaBuffer mPathStates = {};		// (O.xyz, pathIx)[], (D.xyz, meshIx)[], (throughput, ?)[]
		CudaBuffer mHitData = {};			// (bary.x, bary.y), instIx, primIx, tmin
		CudaBuffer mCountersBuffer = {};

		// Denoiser
		OptixDenoiser mDenoiser;
		CudaBuffer mDenoiserScratch = {};
		CudaBuffer mDenoiserState = {};
		CudaBuffer mDenoisedBuffer = {};

		bool mDenoisingEnabled = false;
		bool mDenoisedFrame = false;
		int32_t mDenoiserSampleTreshold = 10;

		// Launch parameters
		LaunchParams mLaunchParams = {};
		CudaBuffer mLaunchParamsBuffer = {};

		// CUDA device properties
		CUcontext mCudaContext = nullptr;
		CUstream mStream = nullptr;
		cudaDeviceProp mDeviceProperties = {};

		// Optix module
		OptixModule mModule = nullptr;
		OptixModuleCompileOptions mModuleCompileOptions = {};

		// Optix pipeline
		OptixPipeline mPipeline = nullptr;
		OptixPipelineCompileOptions mPipelineCompileOptions = {};
		OptixPipelineLinkOptions mPipelineLinkOptions = {};

		// Optix device context
		OptixDeviceContext mOptixContext = nullptr;

		// Render mode data
		struct RenderModeConfig
		{
			OptixProgramGroup rayGenProgram;
			OptixProgramGroup missProgram;
			OptixProgramGroup hitgroupProgram;

			CudaBuffer rayGenRecordsBuffer = {};
			CudaBuffer missRecordsBuffer = {};
			CudaBuffer hitRecordsBuffer = {};

			OptixShaderBindingTable shaderBindingTable = {};
		};
		std::vector<RenderModeConfig> mRenderModeConfigs;

		// Geometry #TODO: remove
		std::vector<CudaBuffer> mVertexBuffers;
		std::vector<CudaBuffer> mNormalBuffers;
		std::vector<CudaBuffer> mTexcoordBuffers;
		std::vector<CudaBuffer> mIndexBuffers;

		// Meshes
		CudaBuffer mCudaMeshData = {};
		CudaBuffer mCudaModelIndices = {};

		// Materials
		CudaBuffer mCudaMaterialData = {};
		CudaBuffer mCudaMaterialOffsets = {};

		// Textures
		struct CudaTexture
		{
			CudaTexture() = default;
			~CudaTexture();
			explicit CudaTexture(std::shared_ptr<Texture> srcTex);

			cudaArray_t mArray = nullptr;
			cudaTextureObject_t mObject = 0;
		};
		std::unordered_map<std::shared_ptr<Texture>, std::shared_ptr<CudaTexture>> mTextures;

		// Optix scene
		OptixTraversableHandle mSceneRoot = 0;
		CudaBuffer mInstancesBuffer = {};
	};
}
