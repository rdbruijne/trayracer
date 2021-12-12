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
		CudaDevice() = default;
		explicit CudaDevice(int deviceID);
		~CudaDevice();

		int DeviceId() const { return mDeviceId; }
		const CUstream Stream() const { return mStream; }
		const CUcontext Context() const { return mCudaContext; }
		const cudaDeviceProp& DeviceProperties() const { return mDeviceProperties; }

		bool IsCurrent() const;
		void SetCurrent();

	private:
		int mDeviceId;

		// CUDA device properties
		CUstream mStream = nullptr;
		CUcontext mCudaContext = nullptr;
		cudaDeviceProp mDeviceProperties = {};
	};
}
