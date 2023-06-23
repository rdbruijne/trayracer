#include "CudaBuffer.h"

namespace Tracer
{
	CudaBuffer::CudaBuffer(size_t size)
	{
		Alloc(size);
	}



	CudaBuffer::CudaBuffer(const CudaBuffer& buffer)
	{
		Resize(buffer.mSize);
		CUDA_CHECK(cudaMemcpyAsync(mPtr, buffer.mPtr, mSize, cudaMemcpyDeviceToDevice));
	}



	CudaBuffer::CudaBuffer(CudaBuffer&& buffer) noexcept :
		mSize(std::exchange(buffer.mSize, 0ull)),
		mPtr(std::exchange(buffer.mPtr, nullptr))
	{
	}



	CudaBuffer::~CudaBuffer()
	{
		Free();
	}



	CudaBuffer& CudaBuffer::operator = (const CudaBuffer& buffer)
	{
		Resize(buffer.mSize);
		CUDA_CHECK(cudaMemcpyAsync(mPtr, buffer.mPtr, mSize, cudaMemcpyDeviceToDevice));
		return *this;
	}



	CudaBuffer& CudaBuffer::operator = (CudaBuffer&& buffer) noexcept
	{
		mSize = std::exchange(buffer.mSize, 0ull);
		mPtr  = std::exchange(buffer.mPtr, nullptr);
		return *this;
	}



	void CudaBuffer::Alloc(size_t size)
	{
		assert(mPtr == nullptr);
		if(size != 0)
		{
			mSize = size;
			CUDA_CHECK(cudaMalloc(&mPtr, size));
		}
	}



	void CudaBuffer::Resize(size_t size)
	{
		if(mSize != size)
		{
			Free();
			Alloc(size);
		}
	}



	void CudaBuffer::Free()
	{
		if(mPtr)
		{
			CUDA_CHECK(cudaFree(mPtr));
			mPtr = nullptr;
		}
		mSize = 0;
	}



	void CudaBuffer::CopyFrom(int srcDeviceId, CudaBuffer& data)
	{
		assert(mSize == data.Size());

		// current device Id
		int deviceId = -1;
		CUDA_CHECK(cudaGetDevice(&deviceId));

		// copy
		CUDA_CHECK(cudaMemcpyPeerAsync(mPtr, deviceId, data.Ptr(), srcDeviceId, data.Size()));
	}



	void CudaBuffer::CopyFrom(int srcDeviceId, CudaBuffer& data, size_t offset, size_t size)
	{
		assert(mSize >= offset + size);

		// current device Id
		int deviceId = -1;
		CUDA_CHECK(cudaGetDevice(&deviceId));

		// copy
		CUDA_CHECK(cudaMemcpyPeerAsync(Ptr<uint8_t>() + offset, deviceId, data.Ptr<uint8_t>() + offset, srcDeviceId, size));
	}
}
