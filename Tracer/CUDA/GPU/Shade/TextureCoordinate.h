#pragma once

#include "CudaUtility.h"

__global__ void TextureCoordinateKernel(DECLARE_KERNEL_PARAMS)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx >= pathCount)
		return;

	// gather path data
	const float4 O4 = pathStates[jobIdx + (stride * 0)];

	// extract path data
	const int32_t pathIx = __float_as_int(O4.w);
	const int32_t pixelIx = pathIx % (resolution.x * resolution.y);

	// hit data
	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t instIx = hd.y;
	const uint32_t primIx = hd.z;

	// didn't hit anything
	if(primIx == ~0)
		return;

	// fetch intersection info
	const IntersectionAttributes attrib = GetIntersectionAttributes(instIx, primIx, bary);
	accumulator[pixelIx] += make_float4(attrib.texcoordX, attrib.texcoordY, 0, 0);
}
