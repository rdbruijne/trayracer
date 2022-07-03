#pragma once

// Project
#include "Common/CommonStructs.h"
#include "CUDA/CudaBuffer.h"
#include "CUDA/CudaTimeEvent.h"
#include "Renderer/RenderStatistics.h"

// CUDA
#include <cuda.h>

// C++
#include <array>
#include <memory>

namespace Tracer
{
	class CameraNode;
	class CudaDevice;
	class OptixRenderer;
	class DeviceRenderer
	{
	public:
		// settings
		static constexpr int MaxTraceDepth = 16;

		explicit DeviceRenderer(std::shared_ptr<CudaDevice> device);
		~DeviceRenderer() = default;

		// reset accumulation
		void Reset() { mLaunchParams.sampleCount = 0; }

		// device
		std::shared_ptr<CudaDevice> Device() { return mDevice; }
		OptixRenderer* Optix() { return mOptixRenderer.get(); }

		// camera
		void SetCamera(CameraNode& camNode);

		// resizing
		void Resize(const int2& resolution);

		// rendering
		RayPickResult PickRay(const int2& pixelIndex);
		void RenderFrame(const KernelSettings& kernelSettings, const RenderModes& renderMode, int firstRow, int rowCount);

		// statistics
		uint32_t SampleCount() const { return mLaunchParams.sampleCount; }
		int2 Resolution() const { return mResolution; }
		const RenderStatistics::DeviceStatistics& Statistics() const { return mRenderStats; }

		// buffers
		const CudaBuffer& Accumulator() const { return mAccumulator; }
		const CudaBuffer& AlbedoBuffer() const { return mAlbedoBuffer; }
		const CudaBuffer& NormalBuffer() const { return mNormalBuffer; }
		const CudaBuffer& ColorBuffer() const { return mColorBuffer; }

	private:
		// render
		void PreRender();
		void PostRender();
		void RenderBounce(int firstRow, int rowCount, int pathLength, uint32_t& pathCount);

		// device
		std::shared_ptr<CudaDevice> mDevice = nullptr;

		// Optix renderer
		std::unique_ptr<OptixRenderer> mOptixRenderer = nullptr;

		// render resolution
		int2 mResolution;

		// timing
		CudaTimeEvent mRenderTimeEvent = {};
		std::array<CudaTimeEvent, MaxTraceDepth> mTraceTimeEvents = {};
		std::array<CudaTimeEvent, MaxTraceDepth> mShadeTimeEvents = {};
		std::array<CudaTimeEvent, MaxTraceDepth> mShadowTimeEvents = {};

		// stats
		RenderStatistics::DeviceStatistics mRenderStats = {};

		// Launch parameters
		LaunchParams mLaunchParams = {};
		CudaBuffer mLaunchParamsBuffer = {};

		// Render buffer
		CudaBuffer mAccumulator = {};
		CudaBuffer mAlbedoBuffer = {};
		CudaBuffer mNormalBuffer = {};
		CudaBuffer mColorBuffer = {};

		// SPT
		CudaBuffer mPathStates = {};		// (O.xyz, pathIx)[], (D.xyz, onSensor)[], (throughput, pdf)[]
		CudaBuffer mHitData = {};			// ((bary.x, bary.y), instIx, primIx, tmin)[]
		CudaBuffer mShadowRayData = {};		// (O.xyz, pixelIx)[], (D.xyz, dist)[], (radiance, ?)[]
		CudaBuffer mCountersBuffer = {};
	};
}
