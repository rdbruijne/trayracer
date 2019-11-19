#include "CudaBuffer.h"

// Project
#include "CudaHelpers.h"

// C++
#include <assert.h>

namespace Tracer
{

	void CudaBuffer::Alloc(size_t size)
	{
		assert(mPtr == nullptr);
		mSize = size;
		const cudaError_t cuResult = cudaMalloc(&mPtr, size);
		Check(cuResult);
	}



	void CudaBuffer::Resize(size_t size)
	{
		if(mPtr)
			Free();
		Alloc(size);
	}



	void CudaBuffer::Free()
	{
		const cudaError_t cuResult = cudaFree(mPtr);
		Check(cuResult);
		mPtr = nullptr;
		mSize = 0;
	}
}
