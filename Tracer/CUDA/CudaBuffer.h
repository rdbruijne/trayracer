#pragma once

// Project
#include "CUDA/CudaError.h"
#include "Utility/Utility.h"

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
		CudaBuffer(const CudaBuffer& buffer);
		CudaBuffer(CudaBuffer&& buffer) noexcept;
		~CudaBuffer();

		// assign
		CudaBuffer& operator = (const CudaBuffer& buffer);
		CudaBuffer& operator = (CudaBuffer&& buffer) noexcept;

		// Memory management
		void Alloc(size_t size);
		void Resize(size_t size);
		void Free();

		// Upload
		template<typename TYPE>
		void Upload(const TYPE* data, size_t count = 1, bool allowResize = false);
		template<typename TYPE>
		void Upload(const std::vector<TYPE>& data, bool allowResize = false);

		template<typename TYPE>
		void UploadAsync(const TYPE* data, size_t count = 1, bool allowResize = false);
		template<typename TYPE>
		void UploadAsync(const std::vector<TYPE>& data, bool allowResize = false);

		// Download
		template<typename TYPE>
		void Download(TYPE* data, size_t count = 1) const;
		template<typename TYPE>
		void Download(std::vector<TYPE>& data) const;

		template<typename TYPE>
		void DownloadAsync(TYPE* data, size_t count = 1) const;
		template<typename TYPE>
		void DownloadAsync(std::vector<TYPE>& data) const;

		// copy from other device
		void CopyFrom(int srcDeviceId, CudaBuffer& data);
		void CopyFrom(int srcDeviceId, CudaBuffer& data, size_t offset, size_t size);

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

#include "CudaBuffer.inl"
