#pragma once

// Project
#include "Cuda/CudaBuffer.h"

// Optix
#include "optix7/optix.h"

namespace Tracer
{
	class Denoiser
	{
	public:
		explicit Denoiser(OptixDeviceContext optixContext);
		~Denoiser();

		void Reset() { mSampleCount = 0; }

		// size
		void Resize(const int2& resolution);

		// enabled
		bool IsEnabled() const { return mEnabled; }
		void SetEnabled(bool enabled) { mEnabled = enabled; }

		// sample count
		uint32_t SampleCount() const { return mSampleCount; }

		// treshold
		uint32_t SampleTreshold() const { return mSampleThreshold; }
		void SetSampleTreshold(uint32_t treshold) { mSampleThreshold = treshold; }

		// buffer
		const CudaBuffer& DenoisedBuffer() const { return mDenoised; }

		// denoise
		void DenoiseFrame(CUstream stream, const int2& resolution, uint32_t sampleCount,
						  const CudaBuffer& colorBuffer, const CudaBuffer& albedoBuffer, const CudaBuffer& normalBuffer);

	private:
		OptixDenoiser mOptixDenoiser;

		// state
		bool mEnabled = false;
		uint32_t mSampleCount = 0;
		uint32_t mSampleThreshold = 10;

		// buffers
		CudaBuffer mScratch = {};
		CudaBuffer mState = {};
		CudaBuffer mDenoised = {};
		CudaBuffer mHdrIntensity = {};
	};
}
