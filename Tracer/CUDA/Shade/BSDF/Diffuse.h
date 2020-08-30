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



static __device__
Closure Diffuse_Closure(ShadingInfo& shadingInfo, const float3& N)
{
	const float u1 = rnd(shadingInfo.seed);
	const float u2 = rnd(shadingInfo.seed);

	const float3 X = make_float3(N.y, -N.x, N.z);
	const float3 Y = N;
	const float3 Z = make_float3(N.x, -N.z, N.y);

	const float3 X3 = make_float3(X.x, Y.x, Z.x);
	const float3 Y3 = make_float3(X.y, Y.y, Z.y);
	const float3 Z3 = make_float3(X.z, Y.z, Z.z);

	const float3 wo = X3 * shadingInfo.wo.x + Y3 * shadingInfo.wo.y + Z3 * shadingInfo.wo.z;
	const float3 wi = X3 * shadingInfo.wi.x + Y3 * shadingInfo.wi.y + Z3 * shadingInfo.wi.z;

	Closure closure = FinalizeClosure(shadingInfo, Diffuse_Eval(wo, wi), Diffuse_Sample(wo, u1, u2));
	closure.extend.wi = X * closure.extend.wi.x + Y * closure.extend.wi.y + Z * closure.extend.wi.z;
	return closure;
}
