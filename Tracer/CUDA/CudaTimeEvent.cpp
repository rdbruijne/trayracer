#include "CudaTimeEvent.h"

// Project
#include "Cuda/CudaError.h"

namespace Tracer
{
	CudaTimeEvent::CudaTimeEvent()
	{
		CUDA_CHECK(cudaEventCreate(&mStart));
		CUDA_CHECK(cudaEventCreate(&mEnd));

		CUDA_CHECK(cudaEventRecord(mStart));
		CUDA_CHECK(cudaEventRecord(mEnd));
	}



	CudaTimeEvent::~CudaTimeEvent()
	{
		CUDA_CHECK(cudaEventDestroy(mStart));
		CUDA_CHECK(cudaEventDestroy(mEnd));
	}



	void CudaTimeEvent::Start(cudaStream_t stream)
	{
		CUDA_CHECK(cudaEventRecord(mStart, stream));
	}



	void CudaTimeEvent::Stop(cudaStream_t stream)
	{
		CUDA_CHECK(cudaEventRecord(mEnd, stream));
	}



	float CudaTimeEvent::Elapsed() const
	{
		float elapsed = 0;
		CUDA_CHECK(cudaEventElapsedTime(&elapsed, mStart, mEnd));
		return elapsed;
	}
}
