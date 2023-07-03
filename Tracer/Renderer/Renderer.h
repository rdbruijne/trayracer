#pragma once

// Project
#include "Common/CommonStructs.h"
#include "CUDA/CudaBuffer.h"
#include "CUDA/CudaTimeEvent.h"
#include "Renderer/DeviceRenderer.h"
#include "Renderer/RenderStatistics.h"
#include "Utility/LinearMath.h"

// Magic Enum
#include "magic_enum/magic_enum.hpp"

// C++
#include <array>
#include <ctime>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace Tracer
{
	class CameraNode;
	class CudaDevice;
	class Denoiser;
	class DeviceRenderer;
	class GLTexture;
	class Material;
	class OptixRenderer;
	class Scene;
	class Texture;
	class Renderer
	{
		// disable copying
		Renderer(const Renderer&) = delete;
		Renderer& operator =(const Renderer&) = delete;

	public:
		// settings
		static constexpr int MaxTraceDepth = 16;

		// functions
		explicit Renderer();
		~Renderer();

		// render
		void UpdateScene(Scene* scene);
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

		inline bool StaticNoise() const { return mUseStaticNoise; }
		void SetStaticNoise(bool useStaticNoise);

		// statistics
		inline RenderStatistics Statistics() const { return mRenderStats; }
		inline uint32_t SampleCount() const { return mDeviceRenderer->SampleCount(); }

		// ray picking
		RayPickResult PickRay(int2 pixelIndex);

		// device
		std::shared_ptr<CudaDevice> Device() { return mCudaDevice; }
		const std::shared_ptr<CudaDevice> Device() const { return mCudaDevice; }

		// device renderer
		std::shared_ptr<DeviceRenderer> DevRenderer() { return mDeviceRenderer; }
		const std::shared_ptr<DeviceRenderer> DevRenderer() const { return mDeviceRenderer; }

		// denoiser
		inline std::shared_ptr<Denoiser> GetDenoiser() { return mDenoiser; }
		inline const std::shared_ptr<Denoiser> GetDenoiser() const { return mDenoiser; }

		// kernel settings
		static KernelSettings DefaultSettings();
		KernelSettings Settings() const { return mKernelSettings; }
		void SetSettings(const KernelSettings& settings)
		{
			mKernelSettings = settings;
			Reset();
		}

	private:
		void Resize(GLTexture* renderTexture);
		bool ShouldDenoise() const;

		// build scene
		void BuildGeometry(Scene* scene);
		void BuildMaterials(Scene* scene);
		void BuildSky(Scene* scene);

		// upload scene
		void UploadGeometry(Scene* scene);
		void UploadMaterials(Scene* scene);
		void UploadSky(Scene* scene);

		// rendering
		void DenoiseFrame();
		void UploadFrame(GLTexture* renderTexture);

		// render mode
		RenderModes mRenderMode = RenderModes::PathTracing;
		MaterialPropertyIds mMaterialPropertyId = MaterialPropertyIds::Diffuse;

		bool mUseStaticNoise = false;
		std::mt19937 mRandomEngine = std::mt19937(static_cast<unsigned int>(std::time(nullptr)));

		// settings
		KernelSettings mKernelSettings = {};

		// Denoiser
		std::shared_ptr<Denoiser> mDenoiser = nullptr;

		// saving
		std::string mSavePath = "";

		// stats
		RenderStatistics mRenderStats = {};

		// timing
		CudaTimeEvent mRenderTimeEvent = {};
		CudaTimeEvent mDenoiseTimeEvent = {};

		// Render buffer
		cudaGraphicsResource* mCudaGraphicsResource = nullptr;

		// CUDA device properties
		std::shared_ptr<CudaDevice> mCudaDevice;
		std::shared_ptr<DeviceRenderer> mDeviceRenderer;

		//
		// build data
		//
		std::vector<std::shared_ptr<Material>> mMaterials;
		std::vector<uint32_t> mModelIndices;
		std::vector<uint32_t> mMaterialOffsets;

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

		// scene rendered last frame
		Scene* mLastScene = nullptr;
	};
}
