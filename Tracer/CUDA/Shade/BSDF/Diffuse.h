#pragma once

#include "Closure.h"



static __device__
float3 Diffuse_SampleHemisphere(float u1, float u2)
{
	const float sinTheta = sqrtf(1.f - u2);
	const float cosTheta = sqrtf(u2);
	const float phi = (2.f * Pi) * u1;
	return make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
}



static __device__
float Diffuse_Pdf()
{
	return 1.f / Pi;
}



static __device__
BsdfResult Diffuse_Eval(const float3& wo, const float3& wi)
{
	BsdfResult result;
	result.wi = wi;
	result.pdf = Diffuse_Pdf();
	result.T = Diffuse_Pdf() * wi.z;
	return result;
}



static __device__
BsdfResult Diffuse_Sample(const float3& wo, float u1, float u2)
{
	return Diffuse_Eval(wo, Diffuse_SampleHemisphere(u1, u2));
}



static __device__
Closure Diffuse_Closure(ShadingInfo& shadingInfo)
{
	const float u1 = rnd(shadingInfo.seed);
	const float u2 = rnd(shadingInfo.seed);
	const BsdfResult eval = Diffuse_Eval(shadingInfo.wo, shadingInfo.wi);
	const BsdfResult sample = Diffuse_Sample(shadingInfo.wo, u1, u2);
	return FinalizeClosure(shadingInfo, eval, sample);
}
