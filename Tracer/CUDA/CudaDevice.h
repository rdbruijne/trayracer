#pragma once

// Project
#include "Renderer/Renderer.h"

// CUDA
#include <cuda.h>

namespace Tracer
{
	class CudaDevice
	{
		friend class Renderer;
	public:
		explicit CudaDevice(int deviceID);
		~CudaDevice();

		int DeviceId() const { return mDeviceId; }
		const CUstream Stream() const { return mStream; }
		const CUcontext Context() const { return mCudaContext; }
		const cudaDeviceProp& DeviceProperties() const { return mDeviceProperties; }

		void MemoryUsage(size_t& free, size_t& total);

		bool IsCurrent() const;
		void SetCurrent();

		static int Count();

	private:
		const int mDeviceId;

		// CUDA device properties
		CUstream mStream = nullptr;
		CUcontext mCudaContext = nullptr;
		cudaDeviceProp mDeviceProperties = {};
	};
}
