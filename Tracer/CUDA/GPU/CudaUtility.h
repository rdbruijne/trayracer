#pragma once

// Project
#include "CudaSky.h"
#include "Common/CommonUtility.h"



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
// Material property
//------------------------------------------------------------------------------------------------------------------------------
static __device__
inline float3 GetColor(const CudaMaterialProperty& prop, float2 uv)
{
	float3 result = make_float3(0);
	if(prop.useColor != 0)
		result = make_float3(__half2float(prop.r), __half2float(prop.g), __half2float(prop.b));
	if(prop.useTexture != 0 && prop.textureMap != 0)
		result *= make_float3(tex2D<float4>(prop.textureMap, uv.x, uv.y));
	return result;
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
// Intersection
//------------------------------------------------------------------------------------------------------------------------------
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
	attrib.matIx = md.materialIndices[primIx] + materialOffsets[modelIx];

	// texcoords
	const float2 uv0 = make_float2(__half2float(tri.uv0x), __half2float(tri.uv0y));
	const float2 uv1 = make_float2(__half2float(tri.uv1x), __half2float(tri.uv1y));
	const float2 uv2 = make_float2(__half2float(tri.uv2x), __half2float(tri.uv2y));
	const float2 texcoord = Barycentric(bary, uv0, uv1, uv2);
	attrib.texcoordX = texcoord.x;
	attrib.texcoordY = texcoord.y;

	// edges
	const float3 v0 = make_float3(tri.v0x, tri.v0y, tri.v0z);
	const float3 v1 = make_float3(tri.v1x, tri.v1y, tri.v1z);
	const float3 v2 = make_float3(tri.v2x, tri.v2y, tri.v2z);
	const float3 e1 = v1 - v0;
	const float3 e2 = v2 - v0;
	attrib.area = length(cross(e1, e2)) * 0.5f;

	// normals
	const float3 N0 = make_float3(__half2float(tri.N0x), __half2float(tri.N0y), __half2float(tri.N0z));
	const float3 N1 = make_float3(__half2float(tri.N1x), __half2float(tri.N1y), __half2float(tri.N1z));
	const float3 N2 = make_float3(__half2float(tri.N2x), __half2float(tri.N2y), __half2float(tri.N2z));
	attrib.shadingNormal = normalize(Barycentric(bary, N0, N1, N2));
	attrib.geometricNormal = make_float3(__half2float(tri.Nx), __half2float(tri.Ny), __half2float(tri.Nz));

	// calculate tangents
	const float s1 = __half2float(tri.uv1x - tri.uv0x);
	const float t1 = __half2float(tri.uv1y - tri.uv0y);

	const float s2 = __half2float(tri.uv2x - tri.uv0x);
	const float t2 = __half2float(tri.uv2y - tri.uv0y);

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
	attrib.diffuse  = GetColor(mat.diffuse, texcoord);
	attrib.emissive = GetColor(mat.emissive, texcoord);
	const float3 normalMap = GetColor(mat.normal, texcoord);

	// apply normal map
	if(normalMap.x > 0.f || normalMap.y > 0.f || normalMap.z > 0.f)
	{
		const float3 norMap = (normalMap * 2.f) - make_float3(1.f);
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
inline float3 SampleLight(uint32_t& seed, const float3& I, const float3& N, float& prob, float& pdf, float3& radiance, float& dist)
{
	// sun energy
	const float3 sunRadiance = SampleSky(skyData->sunDir, false);
	const float sunEnergy = (sunRadiance.x + sunRadiance.y + sunRadiance.z) * skyData->sunArea;
	const float totalEnergy = sunEnergy + lightEnergy;

	// check for any energy
	if(totalEnergy == 0)
	{
		prob = 0;
		pdf = 0;
		radiance = make_float3(0);
		return make_float3(1);
	}

	// try to pick the sun
	if(rnd(seed) * totalEnergy <= sunEnergy)
	{
		prob     = sunEnergy / totalEnergy;
		pdf      = 1.f;
		radiance = sunRadiance;
		dist     = 1e20f;
		return skyData->sunDir;
	}

	// pick random light
	const int32_t lightIx = SelectLight(seed);
	const LightTriangle& tri = lights[lightIx];

	// select point on light
	const float3 bary = make_float3(rnd(seed), rnd(seed), rnd(seed));
	const float3 pointOnLight = (bary.x * tri.V0) + (bary.y * tri.V1) + (bary.z * tri.V2);

	// sample direction (light -> hit)
	float3 L = I - pointOnLight;
	const float sqDist = dot(L, L);
	L = normalize(L);
	const float LNdotL = dot(tri.N, L);
	const float NdotL = dot(N, L);

	// set output parameters
	prob = tri.energy / totalEnergy;
	pdf = (NdotL < 0 && LNdotL > 0) ? sqDist / (tri.area * LNdotL) : 0;
	dist = sqrtf(sqDist);
	radiance = tri.radiance;

	// return hit -> light
	return -L;
}
