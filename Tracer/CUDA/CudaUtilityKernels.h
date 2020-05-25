#pragma once

#include "CudaUtility.h"



//------------------------------------------------------------------------------------------------------------------------------
// Counters
//------------------------------------------------------------------------------------------------------------------------------
static __global__ void InitCudaCountersKernel()
{
	if(threadIdx.x == 0)
	{
		counters->extendRays = 0;
		counters->shadowRays = 0;
	}
}



__host__ void InitCudaCounters()
{
	InitCudaCountersKernel<<<32, 1>>>();
}




//------------------------------------------------------------------------------------------------------------------------------
// Finalize frame
//------------------------------------------------------------------------------------------------------------------------------
__global__ void FimalizeFrameKernel(float4* accumulator, float4* colors, int pixelCount, int sampleCount)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx < pixelCount)
	{
		colors[jobIdx] = accumulator[jobIdx] / sampleCount;
	}
}



__host__ void FinalizeFrame(float4* accumulator, float4* colors, int2 resolution, int sampleCount)
{
	const int32_t pixelCount = resolution.x * resolution.y;
	const uint32_t threadsPerBlock = 128;
	const uint32_t blockCount = DivRoundUp(resolution.x * resolution.y, threadsPerBlock);
	FimalizeFrameKernel<<<blockCount, threadsPerBlock>>>(accumulator, colors, pixelCount, sampleCount);
}
