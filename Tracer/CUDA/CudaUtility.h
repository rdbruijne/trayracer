#pragma once

// Project
#include "Common/CommonStructs.h"
#include "Common/CommonUtility.h"

//------------------------------------------------------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------------------------------------------------------
// math
constexpr float Pi = 3.14159265358979323846f;

// rendering
constexpr float DstMax = 1e30f;
constexpr float Epsilon = 1e-3f;



//------------------------------------------------------------------------------------------------------------------------------
// Globals
//------------------------------------------------------------------------------------------------------------------------------
static __device__ Counters* counters = nullptr;

__constant__ LaunchParams* params = nullptr;

// geometry
__constant__ CudaMeshData* meshData = nullptr;

// materials
__constant__ CudaMatarial* materialData = nullptr;
__constant__ uint32_t* materialOffsets = nullptr;

// instances
__constant__ float4* invInstTransforms = nullptr;
__constant__ uint32_t* modelIndices = nullptr;

// lights
__constant__ int32_t lightCount = 0;
__constant__ float lightEnergy = 0;
__constant__ LightTriangle* lights = nullptr;

// sky
__constant__ SkyData* skyData = nullptr;
__constant__ SkyState* skyStateX = nullptr;
__constant__ SkyState* skyStateY = nullptr;
__constant__ SkyState* skyStateZ = nullptr;



//------------------------------------------------------------------------------------------------------------------------------
// Global setters
//------------------------------------------------------------------------------------------------------------------------------
__host__ void SetCudaCounters(Counters* data)
{
	cudaMemcpyToSymbol(counters, &data, sizeof(void*));
}



__host__ void SetCudaLaunchParams(LaunchParams* data)
{
	cudaMemcpyToSymbol(params, &data, sizeof(void*));
}



// geometry
__host__ void SetCudaMeshData(CudaMeshData* data)
{
	cudaMemcpyToSymbol(meshData, &data, sizeof(void*));
}



// materials
__host__ void SetCudaMatarialData(CudaMatarial* data)
{
	cudaMemcpyToSymbol(materialData, &data, sizeof(void*));
}



__host__ void SetCudaMatarialOffsets(uint32_t* data)
{
	cudaMemcpyToSymbol(materialOffsets, &data, sizeof(void*));
}



// instances
__host__ void SetCudaInvTransforms(float4* data)
{
	cudaMemcpyToSymbol(invInstTransforms, &data, sizeof(void*));
}



__host__ void SetCudaModelIndices(uint32_t* data)
{
	cudaMemcpyToSymbol(modelIndices, &data, sizeof(void*));
}



// lights
__host__ void SetCudaLightCount(int32_t count)
{
	cudaMemcpyToSymbol(lightCount, &count, sizeof(lightCount));
}



__host__ void SetCudaLightEnergy(float energy)
{
	cudaMemcpyToSymbol(lightEnergy, &energy, sizeof(lightEnergy));
}



__host__ void SetCudaLights(LightTriangle* data)
{
	cudaMemcpyToSymbol(lights, &data, sizeof(void*));
}



// sky
__host__ void SetCudaSkyData(SkyData* data)
{
	cudaMemcpyToSymbol(skyData, &data, sizeof(void*));
}



__host__ void SetCudaSkyStateX(SkyState* data)
{
	cudaMemcpyToSymbol(skyStateX, &data, sizeof(void*));
}



__host__ void SetCudaSkyStateY(SkyState* data)
{
	cudaMemcpyToSymbol(skyStateY, &data, sizeof(void*));
}



__host__ void SetCudaSkyStateZ(SkyState* data)
{
	cudaMemcpyToSymbol(skyStateZ, &data, sizeof(void*));
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
// math
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



static __device__
inline float2 DecodeBarycentrics(uint32_t barycentrics)
{
	const uint32_t bx = barycentrics >> 16;
	const uint32_t by = barycentrics & 0xFFFF;
	return make_float2(static_cast<float>(bx) / 65535.f, static_cast<float>(by) / 65535.f);
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
	float area;

	float3 diffuse;
	float pad3;

	float3 emissive;
	float pad4;
};



static __device__
inline IntersectionAttributes GetIntersectionAttributes(uint32_t instIx, uint32_t primIx, float2 bary)
{
	// #TODO: apply instance transform
	IntersectionAttributes attrib = {};

	// fetch triangle
	const CudaMeshData& md = meshData[instIx];
	const PackedTriangle& tri = md.triangles[primIx];

	// set material index
	const uint32_t modelIx = modelIndices[instIx];
	attrib.matIx = tri.matIx + materialOffsets[modelIx];

	// texcoords
	const float2 uv0 = make_float2(tri.uv0x, tri.uv0y);
	const float2 uv1 = make_float2(tri.uv1x, tri.uv1y);
	const float2 uv2 = make_float2(tri.uv2x, tri.uv2y);
	const float2 texcoord = Barycentric(bary, uv0, uv1, uv2);
	attrib.texcoordX = texcoord.x;
	attrib.texcoordY = texcoord.y;

	// edges
	const float3 e1 = tri.v1 - tri.v0;
	const float3 e2 = tri.v2 - tri.v0;
	attrib.area = length(cross(e1, e2)) * 0.5f;

	// normals
	attrib.shadingNormal = normalize(Barycentric(bary, tri.N0, tri.N1, tri.N2));
	attrib.geometricNormal = tri.N;

	// calculate tangents
	const float s1 = tri.uv1x - tri.uv0x;
	const float t1 = tri.uv1y - tri.uv0y;

	const float s2 = tri.uv2x - tri.uv0x;
	const float t2 = tri.uv2y - tri.uv0y;

	const float r = (s1 * t2) - (s2 * t1);

	if(fabsf(r) < 1e-6f)
	{
		attrib.bitangent = normalize(cross(attrib.shadingNormal, e1));
		attrib.tangent   = normalize(cross(attrib.shadingNormal, attrib.bitangent));
	}
	else
	{
		const float rr = 1.f / r;
		const float3 s = ((t2 * e1) - (t1 * e2)) * rr;
		const float3 t = ((s1 * e2) - (s2 * e1)) * rr;

		attrib.bitangent = normalize(t - attrib.shadingNormal * dot(attrib.shadingNormal, t));
		attrib.tangent   = normalize(cross(attrib.shadingNormal, attrib.bitangent));
	}

	// material
	const CudaMatarial& mat = materialData[attrib.matIx];

	// diffuse
	attrib.diffuse = mat.diffuse;
	if(mat.textures & Texture_DiffuseMap)
		attrib.diffuse *= make_float3(tex2D<float4>(mat.diffuseMap, attrib.texcoordX, attrib.texcoordY));

	// emissive
	attrib.emissive = mat.emissive;

	// normal map
	if(mat.textures & Texture_NormalMap)
	{
		const float3 norMap = (make_float3(tex2D<float4>(mat.normalMap, attrib.texcoordX, attrib.texcoordY)) * 2.f) - make_float3(1.f);
		attrib.shadingNormal = normalize(norMap.x * attrib.tangent + norMap.y * attrib.bitangent + norMap.z * attrib.shadingNormal);
	}

	// object -> worldspace
	const float4 tx = invInstTransforms[(instIx * 3) + 0];
	const float4 ty = invInstTransforms[(instIx * 3) + 1];
	const float4 tz = invInstTransforms[(instIx * 3) + 2];

	attrib.shadingNormal   = normalize(transform(tx, ty, tz, attrib.shadingNormal));
	attrib.geometricNormal = normalize(transform(tx, ty, tz, attrib.geometricNormal));
	attrib.bitangent       = normalize(transform(tx, ty, tz, attrib.bitangent));
	attrib.tangent         = normalize(transform(tx, ty, tz, attrib.tangent));

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
	const float phi = 2.f * Pi * v;
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



//------------------------------------------------------------------------------------------------------------------------------
// Next event
//------------------------------------------------------------------------------------------------------------------------------
static __device__
int32_t SelectLight(uint32_t& seed)
{
	const float e = rnd(seed) * lightEnergy;
	int32_t low = 0;
	int32_t high = lightCount - 1;
	while(low <= high)
	{
		const int32_t mid = (low + high) >> 1;
		const LightTriangle& tri = lights[mid];
		if(e < tri.sumEnergy)
			high = mid;
		else if(e > tri.sumEnergy + tri.energy)
			low = mid + 1;
		else
			return mid;
	}

	// failed to find a light using importance sampling, pick a random one from the array
	// #NOTE: we should never get here!
	return clamp((int)(rnd(seed) * lightCount), 0, lightCount - 1);
}



static __device__
inline float3 SampleLight(uint32_t& seed, const float3& I, const float3& N, float& prob, float& pdf, float3& radiance)
{
	if(lightCount == 0)
	{
		prob = 0;
		pdf = 0;
		radiance = make_float3(0);
		return make_float3(1);
	}

	// pick random light
	const int32_t lightIx = SelectLight(seed);
	const LightTriangle& tri = lights[lightIx];

	// select point on light
	const float3 bary = make_float3(rnd(seed), rnd(seed), rnd(seed));
	const float3 pointOnLight = (bary.x * tri.V0) + (bary.y * tri.V1) + (bary.z * tri.V2);

	// sample direction
	float3 L = I - pointOnLight;
	const float sqDist = dot(L, L);
	L = normalize(L);
	const float LNdotL = dot(tri.N, L);
	const float NdotL = dot(N, L);

	prob = tri.energy / lightEnergy;
	pdf = (NdotL < 0 && LNdotL > 0) ? sqDist / (tri.area * LNdotL) : 0;
	radiance = tri.radiance;
	return pointOnLight;
}
