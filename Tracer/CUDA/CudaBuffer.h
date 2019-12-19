#pragma once

// Project
#include "CudaHelpers.h"

// OptiX
#include "Optix7.h"

// C++
#include <assert.h>
#include <vector>

namespace Tracer
{
	class CudaBuffer
	{
	public:
		CudaBuffer() = default;
		CudaBuffer(size_t size);
		~CudaBuffer();

		// Memory management
		void Alloc(size_t size);
		void Resize(size_t size);
		void Free();

		// Upload/download
		template<typename TYPE>
		void Upload(const TYPE* data, size_t count)
		{
			assert(mPtr != nullptr);
			assert(mSize == sizeof(TYPE) * count);
			CUDA_CHECK(cudaMemcpy(mPtr, static_cast<const void*>(data), sizeof(TYPE) * count, cudaMemcpyHostToDevice));
		}

		template<typename TYPE>
		void Download(TYPE* data, size_t count)
		{
			assert(mPtr != nullptr);
			assert(mSize == sizeof(TYPE) * count);
			CUDA_CHECK(cudaMemcpy(static_cast<void*>(data), mPtr, sizeof(TYPE) * count, cudaMemcpyDeviceToHost));
		}

		template<typename TYPE>
		void AllocAndUpload(const std::vector<TYPE>& data)
		{
			Alloc(sizeof(TYPE) * data.size());
			Upload(static_cast<const TYPE*>(data.data()), data.size());
		}

		// inlines
		inline CUdeviceptr DevicePtr() const
		{
			return reinterpret_cast<CUdeviceptr>(mPtr);
		}

		inline const CUdeviceptr* DevicePtrPtr() const
		{
			return reinterpret_cast<const CUdeviceptr*>(&mPtr);
		}

		inline size_t Size() const
		{
			return mSize;
		}

	private:
		size_t mSize = 0;
		void* mPtr = nullptr;
	};
}
