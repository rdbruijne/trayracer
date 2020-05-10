#pragma once

#include "CudaFwd.h"
#include "CudaUtility.h"

static __device__ float2 DecodeBarycentrics(uint32_t barycentrics)
{
	const uint32_t bx = barycentrics >> 16;
	const uint32_t by = barycentrics & 0xFFFF;
	return make_float2(static_cast<float>(bx) / 65535.f, static_cast<float>(by) / 65535.f);
}



static __device__ float3 SampleSky(const float3& O, const float3& D)
{
	return make_float3(1);
}



__global__ void ShadeKernel_AmbientOcclusion(uint32_t pathCount, float4* accumulator, float4* pathStates, uint4* hitData, int2 resolution, uint32_t stride, uint32_t pathLength)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx >= pathCount)
		return;

	// gather data
	const float4 O4 = pathStates[jobIdx + (stride * 0)];
	const float4 D4 = pathStates[jobIdx + (stride * 1)];

	const float3 O = make_float3(O4);
	const float3 D = make_float3(D4);
	const int32_t pathIx = __float_as_int(O4.w);

	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t instIx = hd.y;
	const uint32_t primIx = hd.z;
	const float tmax = __uint_as_float(hd.w);

	if(pathLength == 0)
	{
		if(primIx == ~0)
			return;
		uint32_t seed = tea<2>(pathIx, params->sampleCount + pathLength + 1);

		const IntersectionAttributes attrib = GetIntersectionAttributes(instIx, primIx, bary);
		const float3 newOrigin = O + (D * tmax);
		const float3 newDir = SampleCosineHemisphere(attrib.shadingNormal, rnd(seed), rnd(seed));

		// update path states
		const int32_t extendIx = atomicAdd(&counters->extendRays, 1);
		pathStates[extendIx + (stride * 0)] = make_float4(newOrigin, __int_as_float(pathIx));
		pathStates[extendIx + (stride * 1)] = make_float4(newDir, 0);
	}
	else
	{
		const float z = (tmax > params->aoDist) ? 1.f : tmax / params->aoDist;
		accumulator[pathIx] += make_float4(z, z, z, 0);
	}
}



__global__ void ShadeKernel_DiffuseFilter(uint32_t pathCount, float4* accumulator, float4* pathStates, uint4* hitData, int2 resolution, uint32_t stride, uint32_t pathLength)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx >= pathCount)
		return;

	// gather data
	const float4 O4 = pathStates[jobIdx + (stride * 0)];
	const int32_t pathIx = __float_as_int(O4.w);

	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t instIx = hd.y;
	const uint32_t primIx = hd.z;

	// didn't hit anything
	if(primIx == ~0)
		return;

	const IntersectionAttributes attrib = GetIntersectionAttributes(instIx, primIx, bary);
	const CudaMatarial& mat = materialData[attrib.matIx];

	// diffuse
	float3 diff = mat.diffuse;
	if(mat.textures & Texture_DiffuseMap)
	{
		const float4 diffMap = tex2D<float4>(mat.diffuseMap, attrib.texcoordX, attrib.texcoordY);
		diff *= make_float3(diffMap.z, diffMap.y, diffMap.x);
	}

	accumulator[pathIx] += make_float4(diff, 0);
}



__global__ void ShadeKernel_ObjectID(uint32_t pathCount, float4* accumulator, float4* pathStates, uint4* hitData, int2 resolution, uint32_t stride, uint32_t pathLength)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx >= pathCount)
		return;

	const float4 O4 = pathStates[jobIdx + (stride * 0)];
	const int32_t pathIx = __float_as_int(O4.w);

	// gather data
	const uint4 hd = hitData[pathIx];
	const uint32_t instIx = hd.y;
	const uint32_t primIx = hd.z;

	// didn't hit anything
	if(primIx == ~0)
		return;

	// ID to color
	accumulator[pathIx] += make_float4(IdToColor(instIx + 1), 0);
}



__global__ void ShadeKernel_PathTracing(uint32_t pathCount, float4* accumulator, float4* pathStates, uint4* hitData, int2 resolution, uint32_t stride, uint32_t pathLength)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx >= pathCount)
		return;

	// gather data
	const float4 O4 = pathStates[jobIdx + (stride * 0)];
	const float4 D4 = pathStates[jobIdx + (stride * 1)];
	const float4 T4 = pathLength == 0 ? make_float4(1) : pathStates[jobIdx + (stride * 2)];

	const float3 O = make_float3(O4);
	const float3 D = make_float3(D4);
	const float3 T = make_float3(T4);
	const int32_t pathIx = __float_as_int(O4.w);

	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t instIx = hd.y;
	const uint32_t primIx = hd.z;
	const float tmax = __uint_as_float(hd.w);

	// didn't hit anything
	if(primIx == ~0)
	{
		accumulator[pathIx] += make_float4(T * SampleSky(O, D));
		return;
	}

	// shading
	uint32_t seed = tea<2>(pathIx, params->sampleCount + pathLength + 1);

	const IntersectionAttributes attrib = GetIntersectionAttributes(instIx, primIx, bary);
	const CudaMatarial& mat = materialData[attrib.matIx];

	// diffuse
	float3 diff = mat.diffuse;
	if(mat.textures & Texture_DiffuseMap)
	{
		const float4 diffMap = tex2D<float4>(mat.diffuseMap, attrib.texcoordX, attrib.texcoordY);
		diff *= make_float3(diffMap.z, diffMap.y, diffMap.x);
	}

	// new throughput
	float3 throughput = T * diff;

	// Russian roulette
	if(pathLength > 0)
	{
		const float rr = min(1.0f, max(throughput.x, max(throughput.y, throughput.z)));
		if(rr < rnd(seed))
			return;
		throughput *= 1.f / rr;
	}

	// generate extend
	const float3 newOrigin = O + (D * tmax);
	const float3 newDir = SampleCosineHemisphere(attrib.shadingNormal, rnd(seed), rnd(seed));

	// update path states
	const int32_t extendIx = atomicAdd(&counters->extendRays, 1);
	pathStates[extendIx + (stride * 0)] = make_float4(newOrigin, __int_as_float(pathIx));
	pathStates[extendIx + (stride * 1)] = make_float4(newDir, 0);
	pathStates[extendIx + (stride * 2)] = make_float4(throughput, 0);
}



__global__ void ShadeKernel_ShadingNormal(uint32_t pathCount, float4* accumulator, float4* pathStates, uint4* hitData, int2 resolution, uint32_t stride, uint32_t pathLength)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx >= pathCount)
		return;

	// gather data
	const float4 O4 = pathStates[jobIdx + (stride * 0)];
	const int32_t pathIx = __float_as_int(O4.w);

	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t instIx = hd.y;
	const uint32_t primIx = hd.z;

	// didn't hit anything
	if(primIx == ~0)
		return;

	const IntersectionAttributes attrib = GetIntersectionAttributes(instIx, primIx, bary);
	accumulator[pathIx] += make_float4((attrib.shadingNormal + make_float3(1)) * 0.5f, 0);
}



__global__ void ShadeKernel_TextureCoordinate(uint32_t pathCount, float4* accumulator, float4* pathStates, uint4* hitData, int2 resolution, uint32_t stride, uint32_t pathLength)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx >= pathCount)
		return;

	// gather data
	const float4 O4 = pathStates[jobIdx + (stride * 0)];
	const int32_t pathIx = __float_as_int(O4.w);

	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t instIx = hd.y;
	const uint32_t primIx = hd.z;

	// didn't hit anything
	if(primIx == ~0)
		return;

	const IntersectionAttributes attrib = GetIntersectionAttributes(instIx, primIx, bary);
	accumulator[pathIx] += make_float4(attrib.texcoordX, attrib.texcoordY, 0, 0);
}



__global__ void ShadeKernel_Wireframe(uint32_t pathCount, float4* accumulator, float4* pathStates, uint4* hitData, int2 resolution, uint32_t stride, uint32_t pathLength)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx >= pathCount)
		return;

	// gather data
	const float4 O4 = pathStates[jobIdx + (stride * 0)];

	const float3 O = make_float3(O4);
	const int32_t pathIx = __float_as_int(O4.w);

	const uint4 hd = hitData[pathIx];
	const uint32_t primIx = hd.z;

	if(pathLength == 0)
	{
		float3 newOrigin, newDir;
		uint32_t seed = tea<2>(pathIx, params->sampleCount + pathLength + 1);
		GenerateCameraRay(newOrigin, newDir, make_int2(pathIx % params->resX, pathIx / params->resX), seed);

		// update path states
		const int32_t extendIx = atomicAdd(&counters->extendRays, 1);
		pathStates[extendIx + (stride * 0)] = make_float4(newOrigin, __int_as_float(pathIx));
		pathStates[extendIx + (stride * 1)] = make_float4(newDir, 0);
		pathStates[extendIx + (stride * 2)] = make_float4(__uint_as_float(primIx));
	}
	else
	{
		const float4 T4 = pathStates[jobIdx + (stride * 2)];
		const uint32_t prevT = __float_as_uint(T4.w);
		if(prevT == primIx)
			accumulator[pathIx] += make_float4(1, 1, 1, 0);
	}
}



__global__ void ShadeKernel_ZDepth(uint32_t pathCount, float4* accumulator, float4* pathStates, uint4* hitData, int2 resolution, uint32_t stride, uint32_t pathLength)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx >= pathCount)
		return;

	// gather data
	const float4 O4 = pathStates[jobIdx + (stride * 0)];
	const float4 D4 = pathStates[jobIdx + (stride * 1)];
	const float4 T4 = pathLength == 0 ? make_float4(1) : pathStates[jobIdx + (stride * 2)];

	const float3 O = make_float3(O4);
	const float3 D = make_float3(D4);
	const float3 T = make_float3(T4);
	const int32_t pathIx = __float_as_int(O4.w);

	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t primIx = hd.z;
	const float tmax = __uint_as_float(hd.w);

	// didn't hit anything
	if(primIx == ~0)
	{
		accumulator[pathIx] += make_float4(T * SampleSky(O, D));
		return;
	}

	const float z = tmax * dot(D, params->cameraForward) / params->zDepthMax;
	accumulator[pathIx] += make_float4(z, z, z, 0);
}



__host__ void Shade(RenderModes renderMode, uint32_t pathCount, float4* accumulator, float4* pathStates, uint4* hitData, int2 resolution, uint32_t stride, uint32_t pathLength)
{
	const uint32_t threadsPerBlock = 128;
	const uint32_t blockCount = DivRoundUp(pathCount, threadsPerBlock);
	switch(renderMode)
	{
	case RenderModes::AmbientOcclusion:
		ShadeKernel_AmbientOcclusion<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, resolution, stride, pathLength);
		break;

	case RenderModes::DiffuseFilter:
		ShadeKernel_DiffuseFilter<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, resolution, stride, pathLength);
		break;

	case RenderModes::ObjectID:
		ShadeKernel_ObjectID<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, resolution, stride, pathLength);
		break;

	case RenderModes::PathTracing:
		ShadeKernel_PathTracing<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, resolution, stride, pathLength);
		break;

	case RenderModes::ShadingNormal:
		ShadeKernel_ShadingNormal<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, resolution, stride, pathLength);
		break;

	case RenderModes::TextureCoordinate:
		ShadeKernel_TextureCoordinate<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, resolution, stride, pathLength);
		break;

	case RenderModes::Wireframe:
		ShadeKernel_Wireframe<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, resolution, stride, pathLength);
		break;

	case RenderModes::ZDepth:
		ShadeKernel_ZDepth<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, resolution, stride, pathLength);
		break;

	default:
		break;
	}
}
