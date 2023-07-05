#pragma once

// Project
#include "CudaGlobals.h"
#include "CudaLinearMath.h"
#include "CudaUtility.h"

//------------------------------------------------------------------------------------------------------------------------------
// Intersection data
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



//------------------------------------------------------------------------------------------------------------------------------
// Material data for intersection
//------------------------------------------------------------------------------------------------------------------------------
struct HitMaterial
{
	// #TODO: Pack material properties
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



//------------------------------------------------------------------------------------------------------------------------------
// CudaMaterial properties
//------------------------------------------------------------------------------------------------------------------------------
static __inline__ __device__
int ColorChannels(const CudaMatarial& mat, MaterialPropertyIds propId)
{
	const CudaMaterialProperty& prop = mat.properties[static_cast<size_t>(propId)];
	return prop.colorChannels;
}



static __inline__ __device__
bool HasTexture(const CudaMatarial& mat, MaterialPropertyIds propId)
{
	const CudaMaterialProperty& prop = mat.properties[static_cast<size_t>(propId)];
	return prop.useTexture != 0 && prop.textureMap != 0;
}



static __inline__ __device__
float3 GetColor(const CudaMatarial& mat, MaterialPropertyIds propId, const float2& uv)
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
// Intersection
//------------------------------------------------------------------------------------------------------------------------------
static __inline__ __device__
void FixNormals(Intersection& intersection, const float3& D)
{
	// No transparency (yet): flip fix backfacing normals
	if(dot(D, intersection.shadingNormal) > 0)
	{
		intersection.geometricNormal = -intersection.geometricNormal;
		intersection.shadingNormal = -intersection.shadingNormal;
		intersection.tangent = -intersection.tangent;
		intersection.bitangent = -intersection.bitangent;
	}
}



static __inline__ __device__
void GetIntersectionAttributes(uint32_t instIx, uint32_t primIx, float2 bary, Intersection& intersection, HitMaterial& hitMaterial)
{
	// fetch triangle
	const CudaMeshData& md = MeshData[instIx];
	const PackedTriangle& tri = md.triangles[primIx];

	// set material index
	const uint32_t modelIx = ModelIndices[instIx];
	intersection.matIx = md.materialIndices[primIx] + MaterialOffsets[modelIx];

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

	// material
	const CudaMatarial& mat   = MaterialData[intersection.matIx];

	hitMaterial.diffuse        = GetColor(mat, MaterialPropertyIds::Diffuse, texcoord);
	const float3 tintXYZ       = LinearToCIEXYZ(hitMaterial.diffuse);

	hitMaterial.metallic       = GetColor(mat, MaterialPropertyIds::Metallic, texcoord).x;
	hitMaterial.emissive       = GetColor(mat, MaterialPropertyIds::Emissive, texcoord) * hitMaterial.diffuse;
	hitMaterial.subsurface     = GetColor(mat, MaterialPropertyIds::Subsurface, texcoord).x;
	hitMaterial.tint           = tintXYZ.y > 0 ? CIEXYZToLinear(tintXYZ * (1.f / tintXYZ.y)) : make_float3(1);
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

	// object -> worldspace
	const float4 tx = InvInstTransforms[(instIx * 3) + 0];
	const float4 ty = InvInstTransforms[(instIx * 3) + 1];
	const float4 tz = InvInstTransforms[(instIx * 3) + 2];

	intersection.shadingNormal   = normalize(transform(tx, ty, tz, intersection.shadingNormal));
	intersection.geometricNormal = normalize(transform(tx, ty, tz, intersection.geometricNormal));
	intersection.bitangent       = normalize(transform(tx, ty, tz, intersection.bitangent));
	intersection.tangent         = normalize(transform(tx, ty, tz, intersection.tangent));
}
