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
	/*!
	 * @brief CUDA buffer wrapper.
	 */
	class CudaBuffer
	{
	public:
		/*!
		 * @brief Allocate to given size.
		 * @param[in] size Size in bytes.
		 */
		void Alloc(size_t size);

		/*!
		 * @brief Resize to given size.
		 * @param[in] size Size in bytes.
		 */

		void Resize(size_t size);

		/*!
		 * @brief Free allocated memory.
		 */
		void Free();

		/*!
		 * @brief Upload to buffer.
		 * @param[in] data
		 * @param[in] count Number of elements.
		 */
		template<typename TYPE>
		void Upload(const TYPE* data, size_t count)
		{
			assert(mPtr != nullptr);
			assert(mSize == sizeof(TYPE) * count);
			const cudaError_t cuResult = cudaMemcpy(mPtr, static_cast<const void*>(data), sizeof(TYPE) * count, cudaMemcpyHostToDevice);
			Check(cuResult);
		}

		/*!
		 * @brief Download from buffer.
		 * @param[out] data
		 * @param[in] count Number of elements.
		 */
		template<typename TYPE>
		void Download(TYPE* data, size_t count)
		{
			assert(mPtr != nullptr);
			assert(mSize == sizeof(TYPE) * count);
			const cudaError_t cuResult = cudaMemcpy(static_cast<void*>(data), mPtr, sizeof(TYPE) * count, cudaMemcpyDeviceToHost);
			Check(cuResult);
		}

		/*!
		 * @brief Shorthand for allocating & uploading in a single call.
		 * @param[in] data Vector to upload.
		 */
		template<typename TYPE>
		void AllocAndUpload(const std::vector<TYPE>& data)
		{
			Alloc(sizeof(TYPE) * data.size());
			Upload(static_cast<const TYPE*>(data.data()), data.size());
		}

		/*!
		 * @brief Get CUDA device pointer.
		 * @return CUDA device pointer.
		 */
		inline CUdeviceptr DevicePtr() const
		{
			return reinterpret_cast<CUdeviceptr>(mPtr);
		}

		/*!
		 * @brief Get the buffer's size (in bytes).
		 * @return Size in bytes.
		 */
		inline size_t Size() const
		{
			return mSize;
		}

	private:
		/*!
		 * Size in bytes.
		 */
		size_t mSize = 0;

		/*!
		 * Data pointer.
		 */
		void* mPtr = nullptr;
	};
}
