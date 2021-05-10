#pragma once

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

namespace Tracer
{
	class CudaTimeEvent
	{
	public:
		CudaTimeEvent();
		~CudaTimeEvent();

		void Start(cudaStream_t stream = nullptr);
		void Stop(cudaStream_t stream = nullptr);

		float Elapsed() const;

	private:
		cudaEvent_t mStart = nullptr;
		cudaEvent_t mEnd = nullptr;
	};
}
