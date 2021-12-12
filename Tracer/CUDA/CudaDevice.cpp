#include "CudaDevice.h"

// Project
#include "CUDA/CudaError.h"
#include "Utility/Logger.h"

// CUDA
#include <cuda_runtime.h>

namespace Tracer
{
	CudaDevice::CudaDevice(int deviceID) :
		mDeviceId(deviceID)
	{
		// init CUDA device
		CUDA_CHECK(cudaSetDevice(mDeviceId));
		CUDA_CHECK(cudaStreamCreate(&mStream));
		CUDA_CHECK(cudaGetDeviceProperties(&mDeviceProperties, deviceID));
		CUDA_CHECK(cuCtxGetCurrent(&mCudaContext));

		Logger::Info("Initializing %i: %s", deviceID, mDeviceProperties.name);
	}



	CudaDevice::~CudaDevice()
	{
		SetCurrent();
		CUDA_CHECK(cudaStreamDestroy(mStream));
	}



	bool CudaDevice::IsCurrent() const
	{
		int deviceId = -1;
		CUDA_CHECK(cudaGetDevice(&deviceId));
		return deviceId == mDeviceId;
	}



	void CudaDevice::SetCurrent()
	{
		CUDA_CHECK(cudaSetDevice(mDeviceId));
	}
}