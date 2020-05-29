#pragma once

// Project
#include "Renderer/CudaError.h"
#include "Utility/Utility.h"

// C++
#include <assert.h>
#include <vector>

namespace Tracer
{
	class CudaBuffer
	{
	public:
		NO_COPY_ALLOWED(CudaBuffer)

		CudaBuffer() = default;
		CudaBuffer(size_t size);
		~CudaBuffer();

		// Memory management
		void Alloc(size_t size);
		void Resize(size_t size);
		void Free();

		// Upload
		template<typename TYPE>
		void Upload(const TYPE* data, size_t count = 1, bool allowResize = false)
		{
			if(count == 0)
			{
				if(allowResize)
					Free();
			}
			else
			{
				const size_t size = count * sizeof(TYPE);
				if(allowResize && (!mPtr || mSize != size))
					Resize(size);

				assert(mPtr != nullptr);
				assert(mSize == size);
				CUDA_CHECK(cudaMemcpy(mPtr, static_cast<const void*>(data), size, cudaMemcpyHostToDevice));
			}
		}

		template<typename TYPE>
		void Upload(const std::vector<TYPE>& data, bool allowResize = false)
		{
			Upload(data.data(), data.size(), allowResize);
		}

		// Download
		template<typename TYPE>
		void Download(TYPE* data, size_t count = 1) const
		{
			assert(mPtr != nullptr);
			assert(mSize == sizeof(TYPE) * count);
			CUDA_CHECK(cudaMemcpy(static_cast<void*>(data), mPtr, sizeof(TYPE) * count, cudaMemcpyDeviceToHost));
		}

		template<typename TYPE>
		void Download(std::vector<TYPE>& data) const
		{
			Download(data.data(), data.size());
		}

		// members
		inline size_t Size() const { return mSize; }

		template<typename TYPE = void>
		inline TYPE* Ptr() { return reinterpret_cast<TYPE*>(mPtr); }
		template<typename TYPE = void>
		inline const TYPE* Ptr() const { return reinterpret_cast<TYPE*>(mPtr); }

		// device pointer shorthands
		inline CUdeviceptr DevicePtr() const { return reinterpret_cast<CUdeviceptr>(mPtr); }
		inline const CUdeviceptr* DevicePtrPtr() const { return reinterpret_cast<const CUdeviceptr*>(&mPtr); }

	private:
		size_t mSize = 0;
		void* mPtr = nullptr;
	};
}
