#include "CUDA/CudaBuffer.h"

// Project
#include "CUDA/CudaHelpers.h"

// C++
#include <assert.h>

namespace Tracer
{
	CudaBuffer::CudaBuffer(size_t size)
	{
		Alloc(size);
	}



	CudaBuffer::~CudaBuffer()
	{
		Free();
	}



	void CudaBuffer::Alloc(size_t size)
	{
		assert(mPtr == nullptr);
		mSize = size;
		CUDA_CHECK(cudaMalloc(&mPtr, size));
	}



	void CudaBuffer::Resize(size_t size)
	{
		Free();
		Alloc(size);
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
}
