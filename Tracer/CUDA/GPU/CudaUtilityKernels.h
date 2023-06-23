#pragma once

//------------------------------------------------------------------------------------------------------------------------------
// Counters
//------------------------------------------------------------------------------------------------------------------------------
__global__ void InitCudaCountersKernel()
{
	if(threadIdx.x == 0)
	{
		Counters->extendRays = 0;
		Counters->shadowRays = 0;
	}
}



__host__ void InitCudaCounters()
{
	InitCudaCountersKernel<<<32, 1>>>();
}



//------------------------------------------------------------------------------------------------------------------------------
// Finalize frame
//------------------------------------------------------------------------------------------------------------------------------
__global__ void FimalizeFrameKernel(float4* __restrict__ accumulator, float4* __restrict__ colors, int pixelCount, uint32_t sampleCount)
{
	const int jobIx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIx < pixelCount)
	{
		colors[jobIx] = accumulator[jobIx] / sampleCount;
	}
}



__host__ void FinalizeFrame(float4* __restrict__ accumulator, float4* __restrict__ colors, int resX, int resY, uint32_t sampleCount)
{
	const int32_t pixelCount = resX * resY;
	const uint32_t threadsPerBlock = 128;
	const uint32_t blockCount = DivRoundUp(static_cast<uint32_t>(pixelCount), threadsPerBlock);
	FimalizeFrameKernel<<<blockCount, threadsPerBlock>>>(accumulator, colors, pixelCount, sampleCount);
}
