#pragma once

// Project
#include "CudaSky.h"

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
// Material property
//------------------------------------------------------------------------------------------------------------------------------
static __device__
inline int ColorChannels(const CudaMatarial& mat, MaterialPropertyIds propId)
{
	const CudaMaterialProperty& prop = mat.properties[static_cast<size_t>(propId)];
	return prop.colorChannels;
}



static __device__
inline bool HasTexture(const CudaMatarial& mat, MaterialPropertyIds propId)
{
	const CudaMaterialProperty& prop = mat.properties[static_cast<size_t>(propId)];
	return prop.useTexture != 0 && prop.textureMap != 0;
}



static __device__
inline float3 GetColor(const CudaMatarial& mat, MaterialPropertyIds propId, const float2& uv)
{
	const CudaMaterialProperty& prop = mat.properties[static_cast<size_t>(propId)];

	float3 result = make_float3(1);

	// color
	if(prop.colorChannels == 1)
		result = make_float3(__half2float(prop.r));
	else if(prop.colorChannels == 3)
		result = make_float3(__half2float(prop.r), __half2float(prop.g), __half2float(prop.b));

	// texture
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
// Packing
//------------------------------------------------------------------------------------------------------------------------------
static __device__
uint32_t PackNormal(const float3& N)
{
	// https://aras-p.info/texts/CompactNormalStorage.html -> Spheremap Transform
	float2 enc = normalize(make_float2(N)) * sqrtf(-N.z * 0.5f + 0.5f);
	enc = enc * 0.5f + 0.5f;
	return (static_cast<uint32_t>(N.x * 65535.f) << 16) | static_cast<uint32_t>(N.y * 65535.f);
}



static __device__
float3 UnpackNormal(uint32_t N)
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
// Intersection
//------------------------------------------------------------------------------------------------------------------------------
struct Intersection
{
	float3 geometricNormal;
	float texcoordX;

	float3 shadingNormal;
	float texcoordY;

	float3 tangent;
	uint32_t matIx;

	float3 bitangent;
	float area;
};



// #TODO: Pack material properties
struct HitMaterial
{
	float3 diffuse;
	float metallic;

	float3 emissive;
	float subsurface;

	float3 tint;
	float specular;

	float roughness;
	float specularTint;
	float anisotropic;
	float sheen;

	float sheenTint;
	float clearcoat;
	float clearcoatGloss;
	float luminance;
};



static __device__
inline void GetIntersectionAttributes(uint32_t instIx, uint32_t primIx, float2 bary, Intersection& intersection, HitMaterial& hitMaterial)
{
	// #TODO: apply instance transform

	// fetch triangle
	const CudaMeshData& md = meshData[instIx];
	const PackedTriangle& tri = md.triangles[primIx];

	// set material index
	const uint32_t modelIx = modelIndices[instIx];
	intersection.matIx = md.materialIndices[primIx] + materialOffsets[modelIx];

	// texcoords
	const float2 uv0 = make_float2(__half2float(tri.uv0x), __half2float(tri.uv0y));
	const float2 uv1 = make_float2(__half2float(tri.uv1x), __half2float(tri.uv1y));
	const float2 uv2 = make_float2(__half2float(tri.uv2x), __half2float(tri.uv2y));
	const float2 texcoord = Barycentric(bary, uv0, uv1, uv2);
	intersection.texcoordX = texcoord.x;
	intersection.texcoordY = texcoord.y;

	// edges
	const float3 v0 = make_float3(tri.v0x, tri.v0y, tri.v0z);
	const float3 v1 = make_float3(tri.v1x, tri.v1y, tri.v1z);
	const float3 v2 = make_float3(tri.v2x, tri.v2y, tri.v2z);
	const float3 e1 = v1 - v0;
	const float3 e2 = v2 - v0;
	intersection.area = length(cross(e1, e2)) * 0.5f;

	// normals
	const float3 N0 = make_float3(__half2float(tri.N0x), __half2float(tri.N0y), __half2float(tri.N0z));
	const float3 N1 = make_float3(__half2float(tri.N1x), __half2float(tri.N1y), __half2float(tri.N1z));
	const float3 N2 = make_float3(__half2float(tri.N2x), __half2float(tri.N2y), __half2float(tri.N2z));
	intersection.shadingNormal = normalize(Barycentric(bary, N0, N1, N2));
	intersection.geometricNormal = make_float3(__half2float(tri.Nx), __half2float(tri.Ny), __half2float(tri.Nz));

	// calculate tangents
	const float s1 = __half2float(tri.uv1x - tri.uv0x);
	const float t1 = __half2float(tri.uv1y - tri.uv0y);

	const float s2 = __half2float(tri.uv2x - tri.uv0x);
	const float t2 = __half2float(tri.uv2y - tri.uv0y);

	const float r = (s1 * t2) - (s2 * t1);

	if(fabsf(r) < 1e-6f)
	{
		intersection.bitangent = normalize(cross(intersection.shadingNormal, e1));
		intersection.tangent   = normalize(cross(intersection.shadingNormal, intersection.bitangent));
	}
	else
	{
		const float rr = 1.f / r;
		const float3 s = ((t2 * e1) - (t1 * e2)) * rr;
		const float3 t = ((s1 * e2) - (s2 * e1)) * rr;

		intersection.bitangent = normalize(t - intersection.shadingNormal * dot(intersection.shadingNormal, t));
		intersection.tangent   = normalize(cross(intersection.shadingNormal, intersection.bitangent));
	}

	// material
	const CudaMatarial& mat   = materialData[intersection.matIx];

	hitMaterial.diffuse        = GetColor(mat, MaterialPropertyIds::Diffuse, texcoord);
	const float3 tintXYZ       = LinearRGBToCIEXYZ(hitMaterial.diffuse);

	hitMaterial.metallic       = GetColor(mat, MaterialPropertyIds::Metallic, texcoord).x;
	hitMaterial.emissive       = GetColor(mat, MaterialPropertyIds::Emissive, texcoord);
	hitMaterial.subsurface     = GetColor(mat, MaterialPropertyIds::Subsurface, texcoord).x;
	hitMaterial.tint           = tintXYZ.y > 0 ? CIEXYZToLinearRGB(tintXYZ * (1.f / tintXYZ.y)) : make_float3(1);
	hitMaterial.specular       = GetColor(mat, MaterialPropertyIds::Specular, texcoord).x;
	hitMaterial.roughness      = GetColor(mat, MaterialPropertyIds::Roughness, texcoord).x;
	hitMaterial.specularTint   = GetColor(mat, MaterialPropertyIds::SpecularTint, texcoord).x;
	hitMaterial.anisotropic    = GetColor(mat, MaterialPropertyIds::Anisotropic, texcoord).x;
	hitMaterial.sheen          = GetColor(mat, MaterialPropertyIds::Sheen, texcoord).x;
	hitMaterial.sheenTint      = GetColor(mat, MaterialPropertyIds::SheenTint, texcoord).x;
	hitMaterial.clearcoat      = GetColor(mat, MaterialPropertyIds::Clearcoat, texcoord).x;
	hitMaterial.clearcoatGloss = GetColor(mat, MaterialPropertyIds::ClearcoatGloss, texcoord).x;
	hitMaterial.luminance      = tintXYZ.y;

	// apply normal map
	if(HasTexture(mat, MaterialPropertyIds::Normal))
	{
		const float3 normalMap = GetColor(mat, MaterialPropertyIds::Normal, texcoord);
		const float3 norMap = (normalMap * 2.f) - make_float3(1.f);
		intersection.shadingNormal = normalize(norMap.x * intersection.tangent + norMap.y * intersection.bitangent + norMap.z * intersection.shadingNormal);
	}

	// object -> worldspace
	const float4 tx = invInstTransforms[(instIx * 3) + 0];
	const float4 ty = invInstTransforms[(instIx * 3) + 1];
	const float4 tz = invInstTransforms[(instIx * 3) + 2];

	intersection.shadingNormal   = normalize(transform(tx, ty, tz, intersection.shadingNormal));
	intersection.geometricNormal = normalize(transform(tx, ty, tz, intersection.geometricNormal));
	intersection.bitangent       = normalize(transform(tx, ty, tz, intersection.bitangent));
	intersection.tangent         = normalize(transform(tx, ty, tz, intersection.tangent));
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
	ONB(N, T, B);
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
	ONB(N, T, B);
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
	ONB(normal, tangent, bitangent);

	const float3 f = SampleCosineHemisphere(r0, r1);
	return f.x*tangent + f.y*bitangent + f.z*normal;
}



//------------------------------------------------------------------------------------------------------------------------------
// Next event
//------------------------------------------------------------------------------------------------------------------------------
static __device__
float LightPdf(const float3& D, float dst, float lightArea, const float3& lightNormal)
{
	return (dst * dst) / (fabsf(dot(D, lightNormal)) * lightArea);
}



static __device__
float LightPickProbability(float area, const float3& em)
{
	// scene energy
	const float3 sunRadiance = SampleSky(skyData->sunDir, false);
	const float sunEnergy = (sunRadiance.x + sunRadiance.y + sunRadiance.z) * skyData->sunArea;
	const float totalEnergy = sunEnergy + lightEnergy;

	// light energy
	const float3 energy = em * area;
	const float lightEnergy = energy.x + energy.y + energy.z;
	return lightEnergy / totalEnergy;
}



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
	// energy
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
