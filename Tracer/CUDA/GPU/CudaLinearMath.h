#pragma once

#include "CUDA/helper_math.h"

//------------------------------------------------------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------------------------------------------------------
// math
constexpr float Pi               = 3.1415926535897932f;
constexpr float TwoPi            = 6.2831853071795865f; // 2 * Pi
constexpr float FourPi           = 12.566370614359173f; // 4 * Pi
constexpr float HalfPi           = 1.5707963267948966f; // Pi / 2
constexpr float PiOverFour       = 0.7853981633974483f; // Pi / 4
constexpr float RcpPi            = 0.3183098861837907f; // 1 / Pi
constexpr float TwoOverPi        = 0.6366197723675813f; // 2 / Pi
constexpr float FourOverPi       = 1.2732395447351627f; // 4 / Pi
constexpr float RcpHalfPi        = 0.6366197723675813f; // 1 / (Pi/2)
constexpr float RcpTwoPi         = 0.1591549430918953f; // 1 / (2 * Pi) = 0.5 / Pi
constexpr float RcpFourPi        = 0.0795774715459477f; // 1 / (4 * Pi)
constexpr float SqrtPi           = 1.7724538509055160f; // sqrt(Pi)
constexpr float PiSquare         = 9.8696044010893586f; // Pi^2
constexpr float FourPiSquare     = 39.478417604357434f; // 4 * Pi^2
constexpr float RcpPiSquare      = 0.1013211836423378f; // 1 / (Pi^2) = (1 / Pi)^2
constexpr float RcpFourPiSquare  = 0.0253302959105844f; // 1 / (4 * Pi^2)
constexpr float FourOverPiSquare = 0.4052847345693511f; // 1 / (4 * Pi^2)
constexpr float SqrtTwo          = 1.4142135623730950f; // sqrt(2)
constexpr float RcpSqrtTwo       = 0.7071067811865475f; // 1 / sqrt(2) = sqrt(2) / 2
constexpr float SqrtThree        = 1.7320508075688773f; // sqrt(3)
constexpr float GoldenRatio      = 1.6180339887498948f; // (1 + sqrt(5)) / 2
constexpr float Ln10             = 2.3025850929940457f; // ln(10)



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
static inline __device__
float3 expf(const float3& a)
{
	return make_float3(expf(a.x), expf(a.y), expf(a.z));
}



static inline __device__
float mix(float a, float b, float t)
{
	return (a * (1.f - t)) + (b * t);
}



static inline __device__
float3 mix(const float3& a, const float3& b, float t)
{
	return (a * (1.f - t)) + (b * t);
}



static inline __device__
float3 pow(const float3& f, float p)
{
	return make_float3(powf(f.x, p), powf(f.y, p), powf(f.z, p));
}



static inline __device__
float square(float x)
{
	return x * x;
}



//------------------------------------------------------------------------------------------------------------------------------
// float[n] extensions
//------------------------------------------------------------------------------------------------------------------------------
static inline __device__
float2 operator -(const float2& a)
{
	return make_float2(-a.x, -a.y);
}



static inline __device__
float3 operator -(const float3& a)
{
	return make_float3(-a.x, -a.y, -a.z);
}



static inline __device__
float4 operator -(const float4& a)
{
	return make_float4(-a.x, -a.y, -a.z, -a.w);
}



//------------------------------------------------------------------------------------------------------------------------------
// Value ranges
//------------------------------------------------------------------------------------------------------------------------------
static inline __device__
float3 clamp_scaled(const float3& f3, float maxValue)
{
	const float m = fmaxf(f3.x, fmaxf(f3.y, f3.z));
	if(m <= maxValue)
		return f3;

	return f3 * (maxValue / m);
}



static inline __device__
float3 fixnan(const float3& f3)
{
	return isfinite(f3.x + f3.y + f3.z) ? f3 : make_float3(0);
}



//------------------------------------------------------------------------------------------------------------------------------
// Matrix transform
//------------------------------------------------------------------------------------------------------------------------------
static inline __device__
float3 transform(const float4& tx, const float4& ty, const float4& tz, const float3& v)
{
	return make_float3(
		dot(v, make_float3(tx.x, ty.x, tz.x)),
		dot(v, make_float3(tx.y, ty.y, tz.y)),
		dot(v, make_float3(tx.z, ty.z, tz.z)));
}



static inline __device__
float4 transform(const float4& tx, const float4& ty, const float4& tz, const float4& v)
{
	return make_float4(
		dot(v, make_float4(tx.x, ty.x, tz.x, tx.w)),
		dot(v, make_float4(tx.y, ty.y, tz.y, ty.w)),
		dot(v, make_float4(tx.z, ty.z, tz.z, tz.w)),
		v.w);
}
