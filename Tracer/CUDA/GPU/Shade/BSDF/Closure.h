#pragma once

#include "CudaUtility.h"

enum BsdfFlags
{
	None		= 0,
};



struct alignas(32) BsdfResult
{
	float3 wi;
	float pdf;

	float3 T;
	uint32_t flags;
};



struct Closure
{
	struct alignas(32) MaterialBsdfResult
	{
		float3 T;
		float pdf;

		float3 wi;
		uint32_t flags;
	};

	MaterialBsdfResult extend;
	MaterialBsdfResult shadow;
};



struct ShadingInfo
{
	float3 wo;
	float dst;

	float3 wi;
	float dummy1;

	float3 T;
	float dummy2;
};



static __device__
Closure FinalizeClosure(const ShadingInfo& shadingInfo, const BsdfResult& eval, const BsdfResult& sample)
{
	Closure closure;

	// shadow ray
	closure.shadow.T     = eval.T * abs(eval.wi.z) * shadingInfo.T;
	closure.shadow.pdf   = eval.pdf;
	closure.shadow.flags = eval.flags;

	// extend ray
	closure.extend.T     = sample.T * abs(sample.wi.z) * shadingInfo.T;
	closure.extend.pdf   = sample.pdf;
	closure.extend.wi    = sample.wi;
	closure.extend.flags = eval.flags;

	return closure;
}
