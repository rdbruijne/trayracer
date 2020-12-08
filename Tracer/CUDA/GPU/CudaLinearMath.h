#pragma once

#include "CUDA/helper_math.h"

//------------------------------------------------------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------------------------------------------------------
// math
constexpr float Pi = 3.14159265358979323846f;



//------------------------------------------------------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------------------------------------------------------
static __host__
inline uint32_t DivRoundUp(uint32_t a, uint32_t b)
{
	return (a + b - 1) / b;
}



//------------------------------------------------------------------------------------------------------------------------------
// Generic math functions
//------------------------------------------------------------------------------------------------------------------------------
static __device__
inline float3 expf(const float3& a)
{
	return make_float3(expf(a.x), expf(a.y), expf(a.z));
}



static __device__
inline float3 max(const float3& a, const float3& b)
{
	return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}



static __device__
inline float3 min(const float3& a, const float3& b)
{
	return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}



static __device__
inline float mix(float a, float b, float t)
{
	return (a * (1.f - t)) + (b * t);
}



static __device__
inline float3 mix(const float3& a, const float3& b, float t)
{
	return (a * (1.f - t)) + (b * t);
}



static __device__
inline float3 pow(const float3& f, float p)
{
	return make_float3(powf(f.x, p), powf(f.y, p), powf(f.z, p));
}



//------------------------------------------------------------------------------------------------------------------------------
// float[n] extensions
//------------------------------------------------------------------------------------------------------------------------------
static __device__
inline float2 operator -(const float2& a)
{
	return make_float2(-a.x, -a.y);
}



static __device__
inline float3 operator -(const float3& a)
{
	return make_float3(-a.x, -a.y, -a.z);
}



static __device__
inline float4 operator -(const float4& a)
{
	return make_float4(-a.x, -a.y, -a.z, -a.w);
}



//------------------------------------------------------------------------------------------------------------------------------
// Matrix transform
//------------------------------------------------------------------------------------------------------------------------------
static __device__
inline float3 transform(const float4& tx, const float4& ty, const float4& tz, const float3& v)
{
	return make_float3(
		dot(v, make_float3(tx.x, ty.x, tz.x)),
		dot(v, make_float3(tx.y, ty.y, tz.y)),
		dot(v, make_float3(tx.z, ty.z, tz.z)));
}



static __device__
inline float4 transform(const float4& tx, const float4& ty, const float4& tz, const float4& v)
{
	return make_float4(
		dot(v, make_float4(tx.x, ty.x, tz.x, tx.w)),
		dot(v, make_float4(tx.y, ty.y, tz.y, ty.w)),
		dot(v, make_float4(tx.z, ty.z, tz.z, tz.w)),
		v.w);
}
