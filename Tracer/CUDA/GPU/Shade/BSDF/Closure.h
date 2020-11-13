#pragma once

#include "CudaUtility.h"


struct MaterialBsdfResult
{
	float3 T;
	float pdf;

	float3 wi;
};



struct BsdfResult
{
	float3 wi;
	float pdf;

	float T;
};



struct Closure
{
	MaterialBsdfResult extend;
	MaterialBsdfResult shadow;
};



struct ShadingInfo
{
	float3 wo;
	uint32_t seed;

	float3 wi;

	float3 T;
};



static __device__
Closure FinalizeClosure(const ShadingInfo& shadingInfo, const BsdfResult& eval, const BsdfResult& sample)
{
	Closure closure;

	// shadow ray
	closure.shadow.T = make_float3(eval.T * abs(eval.wi.z)) * shadingInfo.T;
	closure.shadow.pdf = eval.pdf;

	// extend ray
	closure.extend.wi = sample.wi;
	closure.extend.T = make_float3(sample.T * abs(sample.wi.z)) * shadingInfo.T;
	closure.extend.pdf = sample.pdf;

	return closure;
}
