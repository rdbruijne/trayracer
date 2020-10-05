#pragma once

// Project
#include "Common/CommonStructs.h"
#include "Renderer/CudaBuffer.h"
#include "Utility/LinearMath.h"

// Magic Enum
#pragma warning(push)
#pragma warning(disable: 4346 5027)
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
		// settings
		static constexpr int MaxTraceDepth = 16;

		// functions
		explicit Renderer();
		~Renderer();

		// render
		void BuildScene(Scene* scene);
		void RenderFrame(GLTexture* renderTexture);

		// camera
		void SetCamera(CameraNode& camNode);
		void SetCamera(const float3& cameraPos, const float3& cameraForward, const float3& cameraUp, float camFov);

		// render mode
		inline RenderModes RenderMode() const { return mRenderMode; }
		void SetRenderMode(RenderModes mode);

		// statistics
		struct RenderStats
		{
			uint64_t pathCount = 0;

			uint64_t primaryPathCount = 0;
			uint64_t secondaryPathCount = 0;
			uint64_t deepPathCount = 0;
			uint64_t shadowRayCount = 0;

			float renderTimeMs = 0;
			float primaryPathTimeMs = 0;
			float secondaryPathTimeMs = 0;
			float deepPathTimeMs = 0;
			float shadowTimeMs = 0;
			float shadeTimeMs = 0;
			float denoiseTimeMs = 0;
		};
		inline RenderStats Statistics() const { return mRenderStats; }
		inline uint32_t SampleCount() const { return mLaunchParams.sampleCount; }

		// ray picking
		RayPickResult PickRay(int2 pixelIndex);

		// device
		const cudaDeviceProp& CudaDeviceProperties() const { return mDeviceProperties; }

		// denoising
		inline bool DenoisingEnabled() const { return mDenoisingEnabled; }
		inline void SetDenoiserEnabled(bool enabled) { mDenoisingEnabled = enabled; }

		inline uint32_t DenoiserSampleThreshold() const { return mDenoiserSampleThreshold; }
		inline void SetDenoiserSampleThreshold(uint32_t Threshold) { mDenoiserSampleThreshold = Threshold; }

		// kernel settings
#define KERNEL_SETTING(type, name, func, reqClear)				\
		inline type func() const { return mLaunchParams.name; }	\
		inline void Set##func(type name)						\
		{														\
			if(name != mLaunchParams.name)						\
			{													\
				mLaunchParams.name = name;						\
				if (reqClear)									\
					mLaunchParams.sampleCount = 0;				\
			}													\
		}

		KERNEL_SETTING(int, multiSample, MultiSample, false);
		KERNEL_SETTING(int, maxDepth, MaxDepth, true);
		KERNEL_SETTING(float, aoDist, AODist, RenderMode() == RenderModes::AmbientOcclusion || RenderMode() == RenderModes::AmbientOcclusionShading);
		KERNEL_SETTING(float, zDepthMax, ZDepthMax, RenderMode() == RenderModes::ZDepth);

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
		void CreateShaderBindingTable();

		// scene building
		void BuildGeometry(Scene* scene);
		void BuildMaterials(Scene* scene);
		void BuildSky(Scene* scene);

		// render mode
		RenderModes mRenderMode = RenderModes::PathTracing;

		// stats
		RenderStats mRenderStats = {};

		// timing
		class TimeEvent
		{
		public:
			TimeEvent();
			~TimeEvent();

			void Start(cudaStream_t stream = nullptr);
			void Stop(cudaStream_t stream = nullptr);

			float Elapsed() const;
		private:
			cudaEvent_t mStart = nullptr;
			cudaEvent_t mEnd = nullptr;
		};

		TimeEvent mRenderTimeEvents = {};
		TimeEvent mDenoiseTimeEvents = {};
		std::array<TimeEvent, MaxTraceDepth> mTraceTimeEvents = {};
		std::array<TimeEvent, MaxTraceDepth> mShadeTimeEvents = {};
		std::array<TimeEvent, MaxTraceDepth> mShadowTimeEvents = {};

		// Render buffer
		CudaBuffer mAccumulator = {};
		CudaBuffer mAlbedoBuffer = {};
		CudaBuffer mNormalBuffer = {};
		CudaBuffer mColorBuffer = {};
		GLTexture* mRenderTarget = nullptr;
		cudaGraphicsResource* mCudaGraphicsResource = nullptr;

		// SPT
		CudaBuffer mPathStates = {};		// (O.xyz, pathIx)[], (D.xyz, meshIx)[], (throughput, pdf)[]
		CudaBuffer mHitData = {};			// ((bary.x, bary.y), instIx, primIx, tmin)[]
		CudaBuffer mShadowRayData = {};		// (O.xyz, pixelIx)[], (D.xyz, dist)[], (radiance, ?)[]
		CudaBuffer mCountersBuffer = {};

		// Denoiser
		OptixDenoiser mDenoiser;
		CudaBuffer mDenoiserScratch = {};
		CudaBuffer mDenoiserState = {};
		CudaBuffer mDenoisedBuffer = {};
		CudaBuffer mDenoiserHdrIntensity = {};

		bool mDenoisingEnabled = false;
		bool mDenoisedFrame = false;
		uint32_t mDenoiserSampleThreshold = 10;

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

		// Programs
		OptixProgramGroup mRayGenProgram;
		OptixProgramGroup mMissProgram;
		OptixProgramGroup mHitgroupProgram;

		CudaBuffer mRayGenRecordsBuffer = {};
		CudaBuffer mMissRecordsBuffer = {};
		CudaBuffer mHitRecordsBuffer = {};

		// shader bindings
		OptixShaderBindingTable mShaderBindingTable = {};

		// Geometry #TODO: remove
		std::vector<CudaBuffer> mVertexBuffers;
		std::vector<CudaBuffer> mNormalBuffers;
		std::vector<CudaBuffer> mTexcoordBuffers;
		std::vector<CudaBuffer> mIndexBuffers;

		// Meshes
		CudaBuffer mCudaMeshData = {};
		CudaBuffer mCudaModelIndices = {};
		CudaBuffer mCudaInstanceInverseTransforms = {};

		// Materials
		CudaBuffer mCudaMaterialData = {};
		CudaBuffer mCudaMaterialOffsets = {};

		// lights
		CudaBuffer mCudaLightsBuffer = {};

		// sky
		CudaBuffer mSkyData = {};
		CudaBuffer mSkyStateX = {};
		CudaBuffer mSkyStateY = {};
		CudaBuffer mSkyStateZ = {};

		// Optix scene
		OptixTraversableHandle mSceneRoot = 0;
		CudaBuffer mAccelBuffer = {};
		CudaBuffer mInstancesBuffer = {};
	};
}
