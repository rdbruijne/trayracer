#pragma once

__global__ __launch_bounds__(128, 2)
void TangentKernel(DECLARE_KERNEL_PARAMS)
{
	const int jobIx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIx >= pathCount)
		return;

	// gather path data
	const float4 O4 = pathStates[jobIx + (stride * 0)];

	// extract path data
	const uint32_t pathIx = PathIx(__float_as_uint(O4.w));
	const uint32_t pixelIx = pathIx % (resX * resY);

	// hit data
	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t instIx = hd.y;
	const uint32_t primIx = hd.z;

	// didn't hit anything
	if(primIx == ~0)
		return;

	// fetch intersection info
	Intersection intersection = {};
	HitMaterial hitMaterial = {};
	GetIntersectionAttributes(instIx, primIx, bary, intersection, hitMaterial);

	// set result
	accumulator[pixelIx] += make_float4((intersection.tangent + make_float3(1)) * 0.5f, 0);
}
