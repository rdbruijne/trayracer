#pragma once

// Project
#include "CUDA/CudaBuffer.h"
#include "Common/CommonStructs.h"

// Magic Enum
#pragma warning(push)
#pragma warning(disable: 5027)
#include "magic_enum/magic_enum.hpp"
#pragma warning(pop)

// C++
#include <array>
#include <unordered_map>
#include <string>
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
		enum RenderModes
		{
			AmbientOcclusion,
			DiffuseFilter,
			ObjectID,
			PathTracing,
			ShadingNormal,
			TextureCoordinate,
			Wireframe,
			ZDepth
		};

		explicit Renderer();
		~Renderer();

		void BuildScene(Scene* scene);
		void RenderFrame(GLTexture* renderTexture);

		void DownloadPixels(std::vector<float4>& dstPixels);

		void SetCamera(CameraNode& camNode);
		void SetCamera(const float3& cameraPos, const float3& cameraForward, const float3& cameraUp, float camFov);

		void SetRenderMode(RenderModes mode);
		inline RenderModes RenderMode() const { return mRenderMode; }

		inline int SampleCount() const { return mLaunchParams.sampleCount; }

		// ray picking
		RayPickResult PickRay(int2 pixelIndex);

		// denoising
		inline bool DenoisingEnabled() const { return mDenoisingEnabled; }
		inline void SetDenoiserEnabled(bool enabled) { mDenoisingEnabled = enabled; }

		inline int32_t DenoiserSampleTreshold() const { return mDenoiserSampleTreshold; }
		inline void SetDenoiserSampleTreshold(uint32_t treshold) { mDenoiserSampleTreshold = treshold; }

		// kernel settings
#define KERNEL_SETTING(t, n, f)								\
		inline t f() const { return mLaunchParams.n; }		\
		inline void Set##f(t n)								\
		{													\
			if(n != mLaunchParams.n)						\
			{												\
				mLaunchParams.n = n;						\
				mLaunchParams.sampleCount = 0;				\
			}												\
		}

		KERNEL_SETTING(int, maxDepth, MaxDepth)
		KERNEL_SETTING(float, aoDist, AODist)
		KERNEL_SETTING(float, zDepthMax, ZDepthMax)

	private:
		void Resize(const int2& resolution);
		bool ShouldDenoise() const;

		// Creation
		void CreateContext();
		void CreateModule();
		void CreatePrograms();
		void CreatePipeline();
		void CreateDenoiser();

		// scene building
		void BuildGeometry(Scene* scene);
		void BuildShaderBindingTables(Scene* scene);
		void BuildTextures(Scene* scene);

		// render mode
		RenderModes mRenderMode = RenderModes::PathTracing;

		// Render buffer
		CudaBuffer mColorBuffer;
		GLTexture* mRenderTarget = nullptr;
		cudaGraphicsResource* mCudaGraphicsResource = nullptr;

		// Denoiser
		OptixDenoiser mDenoiser;
		CudaBuffer mDenoiserScratch;
		CudaBuffer mDenoiserState;
		CudaBuffer mDenoisedBuffer;

		bool mDenoisingEnabled = false;
		bool mDenoisedFrame = false;
		int32_t mDenoiserSampleTreshold = 10;

		// Launch parameters
		LaunchParams mLaunchParams;
		CudaBuffer mLaunchParamsBuffer;

		// CUDA device properties
		CUcontext mCudaContext;
		CUstream mStream;
		cudaDeviceProp mDeviceProperties;

		// OptiX module
		OptixModule mModule;
		OptixModuleCompileOptions mModuleCompileOptions;

		// OptiX pipeline
		OptixPipeline mPipeline;
		OptixPipelineCompileOptions mPipelineCompileOptions;
		OptixPipelineLinkOptions mPipelineLinkOptions;

		// The OptiX device context
		OptixDeviceContext mOptixContext;

		// Render mode data
		struct RenderModeConfig
		{
			std::vector<OptixProgramGroup> rayGenPrograms;
			std::vector<OptixProgramGroup> missPrograms;
			std::vector<OptixProgramGroup> hitgroupPrograms;

			CudaBuffer rayGenRecordsBuffer;
			CudaBuffer missRecordsBuffer;
			CudaBuffer hitRecordsBuffer;

			OptixShaderBindingTable shaderBindingTable;
		};
		std::array<RenderModeConfig, magic_enum::enum_count<RenderModes>() + 1> mRenderModeConfigs;

		// Geometry
		std::vector<CudaBuffer> mVertexBuffers;
		std::vector<CudaBuffer> mNormalBuffers;
		std::vector<CudaBuffer> mTexcoordBuffers;
		std::vector<CudaBuffer> mIndexBuffers;

		// Textures
		struct OptixTexture
		{
			OptixTexture() = default;
			explicit OptixTexture(std::shared_ptr<Texture> srcTex);
			cudaArray_t mArray = nullptr;
			cudaTextureObject_t mObject = 0;
		};
		std::unordered_map<std::shared_ptr<Texture>, OptixTexture> mTextures;

		// OptiX scene
		OptixTraversableHandle mSceneRoot = 0;
		CudaBuffer mAccelBuffer;
	};

	std::string ToString(Renderer::RenderModes renderMode);
}
