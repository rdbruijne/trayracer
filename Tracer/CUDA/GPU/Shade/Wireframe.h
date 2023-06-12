#pragma once

__global__ __launch_bounds__(128, 2)
void WireframeKernel(DECLARE_KERNEL_PARAMS)
{
	const int jobIx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIx >= pathCount)
		return;

	// gather path data
	const float4 O4 = pathStates[jobIx + (stride * 0)];
	const float4 D4 = pathStates[jobIx + (stride * 1)];

	// extract path data
	const uint32_t pathIx = PathIx(__float_as_uint(O4.w));

	// hit data
	const uint4 hd = hitData[pathIx];
	const uint32_t primIx = hd.z;

	if(pathLength == 0)
	{
		// update path states
		const int32_t extendIx = atomicAdd(&Counters->extendRays, 1);
		pathStates[extendIx + (stride * 0)] = make_float4(O4.x, O4.y, O4.z, __uint_as_float(Pack(pathIx)));
		pathStates[extendIx + (stride * 1)] = make_float4(D4.x, D4.y, D4.z, 0);
		pathStates[extendIx + (stride * 2)] = make_float4(__uint_as_float(primIx));
	}
	else
	{
		const float4 T4 = pathStates[jobIx + (stride * 2)];
		const uint32_t prevT = __float_as_uint(T4.w);
		const int32_t pixelIx = pathIx % (resolution.x * resolution.y);
		if(prevT == primIx)
			accumulator[pixelIx] += make_float4(1, 1, 1, 0);
	}
}
