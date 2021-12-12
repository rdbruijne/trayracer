#pragma once

// Project
#include "Common/CommonStructs.h"
#include "CUDA/CudaBuffer.h"
#include "CUDA/CudaTimeEvent.h"
#include "Renderer/RenderStatistics.h"
#include "Utility/LinearMath.h"

// Magic Enum
#pragma warning(push)
#pragma warning(disable: 4346 5027)
#include "magic_enum/magic_enum.hpp"
#pragma warning(pop)

// C++
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Tracer
{
	class CameraNode;
	class CudaDevice;
	class Denoiser;
	class GLTexture;
	class OptixRenderer;
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
		void Reset();

		// save render
		inline void RequestSave(const std::string& path) { mSavePath = path; }
		inline void ResetSaveRequest() { mSavePath = ""; }
		bool SaveRequested(std::string& path);

		// camera
		void SetCamera(CameraNode& camNode);

		// render mode
		inline RenderModes RenderMode() const { return mRenderMode; }
		void SetRenderMode(RenderModes mode);

		inline MaterialPropertyIds MaterialPropertyId() const { return mMaterialPropertyId; }
		void SetMaterialPropertyId(MaterialPropertyIds id);

		// statistics
		inline RenderStatistics Statistics() const { return mRenderStats; }
		inline uint32_t SampleCount() const { return mLaunchParams.sampleCount; }

		// ray picking
		RayPickResult PickRay(int2 pixelIndex);

		// device
		std::shared_ptr<CudaDevice> Device() { return mCudaDevice; }
		const std::shared_ptr<CudaDevice> Device() const { return mCudaDevice; }

		// denoiser
		inline std::shared_ptr<Denoiser> GetDenoiser() { return mDenoiser; }
		inline const std::shared_ptr<Denoiser> GetDenoiser() const { return mDenoiser; }

		// kernel settings
#define KERNEL_SETTING(type, name, func, reqClear)				\
		inline type func() const { return mLaunchParams.name; }	\
		inline void Set##func(type name)						\
		{														\
			if(name != mLaunchParams.name)						\
			{													\
				mLaunchParams.name = name;						\
				if (reqClear)									\
					Reset();									\
			}													\
		}

		KERNEL_SETTING(int, multiSample, MultiSample, false);
		KERNEL_SETTING(int, maxDepth, MaxDepth, true);
		KERNEL_SETTING(float, aoDist, AODist, RenderMode() == RenderModes::AmbientOcclusion || RenderMode() == RenderModes::AmbientOcclusionShading);
		KERNEL_SETTING(float, zDepthMax, ZDepthMax, RenderMode() == RenderModes::ZDepth);

#undef KERNEL_SETTING

	private:
		void Resize(GLTexture* renderTexture);
		bool ShouldDenoise() const;

		// scene building
		void BuildGeometry(Scene* scene);
		void BuildMaterials(Scene* scene);
		void BuildSky(Scene* scene);

		// rendering
		void PreRenderUpdate();
		void PostRenderUpdate();
		void RenderBounce(int pathLength, uint32_t& pathCount);
		void DenoiseFrame();
		void UploadFrame(GLTexture* renderTexture);

		// render mode
		RenderModes mRenderMode = RenderModes::PathTracing;
		MaterialPropertyIds mMaterialPropertyId = MaterialPropertyIds::Diffuse;

		// Optix renderer
		std::unique_ptr<OptixRenderer> mOptixRenderer = nullptr;

		// Denoiser
		std::shared_ptr<Denoiser> mDenoiser = nullptr;

		// saving
		std::string mSavePath = "";

		// stats
		RenderStatistics mRenderStats = {};

		// timing
		CudaTimeEvent mRenderTimeEvent = {};
		CudaTimeEvent mDenoiseTimeEvent = {};
		std::array<CudaTimeEvent, MaxTraceDepth> mTraceTimeEvents = {};
		std::array<CudaTimeEvent, MaxTraceDepth> mShadeTimeEvents = {};
		std::array<CudaTimeEvent, MaxTraceDepth> mShadowTimeEvents = {};

		// Render buffer
		CudaBuffer mAccumulator = {};
		CudaBuffer mAlbedoBuffer = {};
		CudaBuffer mNormalBuffer = {};
		CudaBuffer mColorBuffer = {};
		cudaGraphicsResource* mCudaGraphicsResource = nullptr;

		// SPT
		CudaBuffer mPathStates = {};		// (O.xyz, pathIx)[], (D.xyz, onSensor)[], (throughput, pdf)[]
		CudaBuffer mHitData = {};			// ((bary.x, bary.y), instIx, primIx, tmin)[]
		CudaBuffer mShadowRayData = {};		// (O.xyz, pixelIx)[], (D.xyz, dist)[], (radiance, ?)[]
		CudaBuffer mCountersBuffer = {};

		// Launch parameters
		LaunchParams mLaunchParams = {};
		CudaBuffer mLaunchParamsBuffer = {};

		// CUDA device properties
		std::shared_ptr<CudaDevice> mCudaDevice;

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
	};
}
