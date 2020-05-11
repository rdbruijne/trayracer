#pragma once

// Project
#include "Common/CommonStructs.h"
#include "Common/CommonUtility.h"

// C++
//#include <stdint.h>

// CUDA
//#include "CUDA/helper_math.h"
//#include "CUDA/random.h"



//------------------------------------------------------------------------------------------------------------------------------
// Cconstants
//------------------------------------------------------------------------------------------------------------------------------
// math
constexpr float M_E        = 2.71828182845904523536f;   // e
constexpr float M_LOG2E    = 1.44269504088896340736f;   // log2(e)
constexpr float M_LOG10E   = 0.434294481903251827651f;  // log10(e)
constexpr float M_LN2      = 0.693147180559945309417f;  // ln(2)
constexpr float M_LN10     = 2.30258509299404568402f;   // ln(10)
constexpr float M_PI       = 3.14159265358979323846f;   // pi
constexpr float M_PI_2     = 1.57079632679489661923f;   // pi/2
constexpr float M_PI_4     = 0.785398163397448309616f;  // pi/4
constexpr float M_1_PI     = 0.318309886183790671538f;  // 1/pi
constexpr float M_2_PI     = 0.636619772367581343076f;  // 2/pi
constexpr float M_2_SQRTPI = 1.12837916709551257390f;   // 2/sqrt(pi)
constexpr float M_SQRT2    = 1.41421356237309504880f;   // sqrt(2)
constexpr float M_SQRT1_2  = 0.707106781186547524401f;  // 1/sqrt(2)

// rendering
constexpr float DST_MAX = 1e30f;



//------------------------------------------------------------------------------------------------------------------------------
// Globals
//------------------------------------------------------------------------------------------------------------------------------
static __device__ Counters* counters;

__constant__ CudaMatarial* materialData;
__constant__ CudaMeshData* meshData;
__constant__ LaunchParams* params;
__constant__ uint32_t* materialOffsets;
__constant__ uint32_t* modelIndices;



__host__ void SetCudaCounters(Counters* data)
{
	cudaMemcpyToSymbol(counters, &data, sizeof(void*));
}



__host__ void SetCudaMatarialData(CudaMatarial* data)
{
	cudaMemcpyToSymbol(materialData, &data, sizeof(void*));
}



__host__ void SetCudaMatarialOffsets(uint32_t* data)
{
	cudaMemcpyToSymbol(materialOffsets, &data, sizeof(void*));
}



__host__ void SetCudaModelIndices(uint32_t* data)
{
	cudaMemcpyToSymbol(modelIndices, &data, sizeof(void*));
}



__host__ void SetCudaMeshData(CudaMeshData* data)
{
	cudaMemcpyToSymbol(meshData, &data, sizeof(void*));
}



__host__ void SetCudaLaunchParams(LaunchParams* data)
{
	cudaMemcpyToSymbol(params, &data, sizeof(void*));
}



//------------------------------------------------------------------------------------------------------------------------------
// Counters
//------------------------------------------------------------------------------------------------------------------------------
static __global__ void InitCudaCountersKernel()
{
	if(threadIdx.x == 0)
	{
		counters->extendRays = 0;
	}
}



__host__ void InitCudaCounters()
{
	InitCudaCountersKernel<<<32, 1>>>();
}




//------------------------------------------------------------------------------------------------------------------------------
// Math
//------------------------------------------------------------------------------------------------------------------------------
static __host__ inline uint32_t DivRoundUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

static __device__ inline float2 operator -(const float2& a) { return make_float2(-a.x, -a.y); }
static __device__ inline float3 operator -(const float3& a) { return make_float3(-a.x, -a.y, -a.z); }
static __device__ inline float4 operator -(const float4& a) { return make_float4(-a.x, -a.y, -a.z, -a.w); }




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



//------------------------------------------------------------------------------------------------------------------------------
// Intersection
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



struct IntersectionAttributes
{
	float3 geometricNormal;
	float texcoordX;

	float3 shadingNormal;
	float texcoordY;

	float3 tangent;
	uint32_t matIx;

	float3 bitangent;
	int pad2;
};



static __device__
inline IntersectionAttributes GetIntersectionAttributes(uint32_t instIx, uint32_t primIx, float2 bary)
{
	IntersectionAttributes attrib = {};

	// get optix info
	const CudaMeshData& md = meshData[instIx];
	const uint3 index = md.indices[primIx];

	// set material index
	const uint32_t modelIx = modelIndices[instIx];
	attrib.matIx = md.matIndices[primIx] + materialOffsets[modelIx];

	// set simple data
	attrib.shadingNormal = normalize(Barycentric(bary, md.normals[index.x], md.normals[index.y], md.normals[index.z]));

	const float2 texcoord0 = md.texcoords[index.x];
	const float2 texcoord1 = md.texcoords[index.y];
	const float2 texcoord2 = md.texcoords[index.z];
	const float2 texcoord = Barycentric(bary, texcoord0, texcoord1, texcoord2);;
	attrib.texcoordX = texcoord.x;
	attrib.texcoordY = texcoord.y;

	// calculate geometric normal
	const float3 v0 = md.vertices[index.x];
	const float3 v1 = md.vertices[index.y];
	const float3 v2 = md.vertices[index.z];

	const float3 e1 = v1 - v0;
	const float3 e2 = v2 - v0;
	const float3 N = cross(e1, e2);
	attrib.geometricNormal = normalize(N);

	// calculate tangents
	const float s1 = texcoord1.x - texcoord0.x;
	const float t1 = texcoord1.y - texcoord0.y;

	const float s2 = texcoord2.x - texcoord0.x;
	const float t2 = texcoord2.y - texcoord0.y;

	float r = (s1 * t2) - (s2 * t1);

	if(fabsf(r) < 1e-4f)
	{
		attrib.bitangent = normalize(cross(attrib.shadingNormal, e1));
		attrib.tangent = normalize(cross(attrib.shadingNormal, attrib.bitangent));
	}
	else
	{
		r = 1.f / r;
		const float3 t = make_float3((s1 * e2.x - s2 * e1.x) * r, (s1 * e2.y - s2 * e1.y) * r, (s1 * e2.z - s2 * e1.z) * r);
		attrib.bitangent = normalize(t - attrib.shadingNormal * dot(attrib.shadingNormal, t));
		attrib.tangent = normalize(cross(attrib.shadingNormal, attrib.bitangent));
	}

	return attrib;
}



//------------------------------------------------------------------------------------------------------------------------------
// Ray
//------------------------------------------------------------------------------------------------------------------------------
static __device__
inline void GenerateCameraRay(float3& O, float3& D, int2 pixelIndex, uint32_t& seed)
{
	GenerateCameraRay(params->cameraPos, params->cameraForward, params->cameraSide, params->cameraUp, params->cameraFov,
					   make_float2(params->resX, params->resY), O, D, pixelIndex, seed);
}



//------------------------------------------------------------------------------------------------------------------------------
// Orthonormal base
//------------------------------------------------------------------------------------------------------------------------------
static __device__
inline void ONB(const float3 normal, float3& tangent, float3& bitangent)
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
// Sampling
//------------------------------------------------------------------------------------------------------------------------------
static __device__
inline float3 SampleCosineHemisphere(float u, float v)
{
	// uniform sample disk
	const float r = sqrtf(u);
	const float phi = 2.f * M_PI * v;
	const float x = r * cosf(phi);
	const float y = r * sinf(phi);
	const float z = sqrtf(fmaxf(0.f, 1.f - x*x - y*y));
	return make_float3(x, y, z);
}



static __device__
inline float3 SampleCosineHemisphere(const float3& normal, float u, float v)
{
	float3 tangent, bitangent;
	ONB(normal, tangent, bitangent);

	const float3 f = SampleCosineHemisphere(u, v);
	return f.x*tangent + f.y*bitangent + f.z*normal;
}
