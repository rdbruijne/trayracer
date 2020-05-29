#include "CudaBuffer.h"

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
}
