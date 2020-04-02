#pragma once

// Project
#include "CUDA/CudaBuffer.h"
#include "Common/CommonStructs.h"

// libraries
#include "magic_enum/magic_enum.hpp"

// C++
#include <array>
#include <unordered_map>
#include <string>
#include <vector>

namespace Tracer
{
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

		void SetCamera(float3 cameraPos, float3 cameraForward, float3 cameraUp, float camFov);

		void SetRenderMode(RenderModes mode);
		inline RenderModes RenderMode() const { return mRenderMode; }

		inline int SampleCount() const { return mLaunchParams.sampleCount; }

	private:
		// Creation
		void CreateContext();
		void CreateModule();
		void CreateRaygenPrograms();
		void CreateMissPrograms();
		void CreateHitgroupPrograms();
		void CreatePipeline();

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
		std::array<RenderModeConfig, magic_enum::enum_count<RenderModes>()> mRenderModeConfigs;

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
