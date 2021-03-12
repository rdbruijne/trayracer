#pragma once

// CUDA
#include "CUDA/random.h"

//------------------------------------------------------------------------------------------------------------------------------
// Colors
//------------------------------------------------------------------------------------------------------------------------------
// Convert an object ID to a color
static __device__
inline float3 IdToColor(uint32_t id)
{
	// https://stackoverflow.com/a/9044057
	uint32_t c[3] = { 0, 0, 0 };
	for(uint32_t i = 0; i < 3; i++)
	{
		c[i] = (id >> i) & 0x249249;
		c[i] = ((c[i] << 1) | (c[i] >>  3)) & 0x0C30C3;
		c[i] = ((c[i] << 2) | (c[i] >>  6)) & 0x00F00F;
		c[i] = ((c[i] << 4) | (c[i] >> 12)) & 0x0000FF;
	}

	return make_float3(c[0] * (1.f / 255.f), c[1] * (1.f / 255.f), c[2] * (1.f / 255.f));
}



static __device__
inline float3 LinearRGBToCIEXYZ(const float3& rgb)
{
	return make_float3(
		max(0.0f, 0.412453f * rgb.x + 0.357580f * rgb.y + 0.180423f * rgb.z),
		max(0.0f, 0.212671f * rgb.x + 0.715160f * rgb.y + 0.072169f * rgb.z),
		max(0.0f, 0.019334f * rgb.x + 0.119193f * rgb.y + 0.950227f * rgb.z));
}



static __device__
inline float3 CIEXYZToLinearRGB(const float3& xyz)
{
	return make_float3(
		max(0.0f,  3.240479f * xyz.x - 1.537150f * xyz.y - 0.498535f * xyz.z),
		max(0.0f, -0.969256f * xyz.x + 1.875992f * xyz.y + 0.041556f * xyz.z),
		max(0.0f,  0.055648f * xyz.x - 0.204043f * xyz.y + 1.057311f * xyz.z));
}



//------------------------------------------------------------------------------------------------------------------------------
// Barycentrics
//------------------------------------------------------------------------------------------------------------------------------
static __device__
inline float2 Barycentric(float2 bc, const float2& v0, const float2& v1, const float2& v2)
{
	return v0 + ((v1 - v0) * bc.x) + ((v2 - v0) * bc.y);
}



static __device__
inline float3 Barycentric(float2 bc, const float3& v0, const float3& v1, const float3& v2)
{
	return v0 + ((v1 - v0) * bc.x) + ((v2 - v0) * bc.y);
}



static __device__
inline float2 DecodeBarycentrics(uint32_t barycentrics)
{
	const uint32_t bx = barycentrics >> 16;
	const uint32_t by = barycentrics & 0xFFFF;
	return make_float2(static_cast<float>(bx) / 65535.f, static_cast<float>(by) / 65535.f);
}



//------------------------------------------------------------------------------------------------------------------------------
// Packing
//------------------------------------------------------------------------------------------------------------------------------
static __device__
inline uint32_t PackNormal(const float3& N)
{
	// https://aras-p.info/texts/CompactNormalStorage.html -> Spheremap Transform
	float2 enc = normalize(make_float2(N)) * sqrtf(-N.z * 0.5f + 0.5f);
	enc = enc * 0.5f + 0.5f;
	return (static_cast<uint32_t>(N.x * 65535.f) << 16) | static_cast<uint32_t>(N.y * 65535.f);
}



static __device__
inline float3 UnpackNormal(uint32_t N)
{
	// https://aras-p.info/texts/CompactNormalStorage.html -> Spheremap Transform
	const float nx = static_cast<float>(N >> 16) / 65535.f;
	const float ny = static_cast<float>(N & 0xFFFF) / 65535.f;
	const float4 nn = make_float4(nx, ny, 0, 0) * make_float4(2, 2, 0, 0) + make_float4(-1, -1, 1, -1);
	const float l = dot(make_float3(nn.x, nn.y, nn.z), make_float3(-nn.x, -nn.y, -nn.w));
	const float sqrl = sqrtf(l);
	return (make_float3(nx * sqrl, ny * sqrl, l) * 2.0f) + make_float3(0, 0, -1);
}



//------------------------------------------------------------------------------------------------------------------------------
// Orthonormal base
//------------------------------------------------------------------------------------------------------------------------------
static __device__
inline void OrthonormalBase(const float3 normal, float3& tangent, float3& bitangent)
{
	if(fabsf(normal.x) > fabsf(normal.y))
	{
		bitangent.x = -normal.y;
		bitangent.y =  normal.x;
		bitangent.z =  0;
	}
	else
	{
		bitangent.x =  0;
		bitangent.y = -normal.z;
		bitangent.z = normal.y;
	}

	bitangent = normalize(bitangent);
	tangent = cross(bitangent, normal);
}



//------------------------------------------------------------------------------------------------------------------------------
// Space transformations
//------------------------------------------------------------------------------------------------------------------------------
static __device__
inline float3 WorldToTangent(const float3& V, const float3& N, const float3& T, const float3& B)
{
	return make_float3(dot(V, T), dot(V, B), dot(V, N));
}



static __device__
inline float3 WorldToTangent(const float3& V, const float3& N)
{
	float3 T, B;
	OrthonormalBase(N, T, B);
	return WorldToTangent(V, N, T, B);
}



static __device__
inline float3 TangentToWorld(const float3& V, const float3& N, const float3& T, const float3& B)
{
	return (V.x * T) + (V.y * B) + (V.z * N);
}



static __device__
inline float3 TangentToWorld(const float3& V, const float3& N)
{
	float3 T, B;
	OrthonormalBase(N, T, B);
	return TangentToWorld(V, N, T, B);
}



//------------------------------------------------------------------------------------------------------------------------------
// Sampling
//------------------------------------------------------------------------------------------------------------------------------
static __device__
inline float3 SampleHemisphere(float r0, float r1)
{
	const float sinTheta = sqrtf(1.f - r1);
	const float cosTheta = sqrtf(r1);
	const float phi = TwoPi * r0;
	return make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
}



static __device__
inline float3 SampleCosineHemisphere(float r0, float r1)
{
	// uniform sample disk
	const float phi = TwoPi * r0;
	float sinPhi, cosPhi;
	sincosf(phi, &sinPhi, &cosPhi);

	const float b = sqrtf(1.f - r1);
	return make_float3(cosPhi * b, sinPhi * b, sqrtf(r1));
}



static __device__
inline float3 SampleCosineHemisphere(const float3& normal, float r0, float r1)
{
	float3 tangent, bitangent;
	OrthonormalBase(normal, tangent, bitangent);

	const float3 f = SampleCosineHemisphere(r0, r1);
	return f.x*tangent + f.y*bitangent + f.z*normal;
}
