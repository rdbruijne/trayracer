#pragma once

// Optix
#include "optix7/optix.h"

#ifndef __CUDACC__
// Magic Enum
#pragma warning(push)
#pragma warning(disable: 4346 5027)
#include "magic_enum/magic_enum.hpp"
#pragma warning(pop)
#endif

// CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// C++
#include <stdint.h>

//------------------------------------------------------------------------------------------------------------------------------
// params
//------------------------------------------------------------------------------------------------------------------------------
#define DECLARE_KERNEL_PARAMS	uint32_t pathCount, float4* accumulator, float4* albedo, float4* normals, float4* pathStates,\
								uint4* hitData, float4* shadowRays, int2 resolution, uint32_t stride, uint32_t pathLength
#define PASS_KERNEL_PARAMS		pathCount, accumulator, albedo, normals, pathStates, hitData, shadowRays, resolution, stride, pathLength



//------------------------------------------------------------------------------------------------------------------------------
// Enumerations
//------------------------------------------------------------------------------------------------------------------------------
enum RayTypes
{
	RayType_Surface = 0,
	RayType_Shadow,

	RayType_Count
};



enum RayGenModes : uint32_t
{
	RayGen_Primary,
	RayGen_Secondary,
	RayGen_Shadow,
	RayGen_RayPick
};



enum TexturesInMaterial
{
	Texture_DiffuseMap  = 0x1,
	Texture_NormalMap   = 0x2
};



enum class RenderModes : uint32_t
{
	AmbientOcclusion,
	AmbientOcclusionShading,
	Bitangent,
	DiffuseFilter,
	DirectLight,
	GeometricNormal,
	MaterialID,
	ObjectID,
	PathTracing,
	ShadingNormal,
	Tangent,
	TextureCoordinate,
	Wireframe,
	ZDepth
};
#ifndef __CUDACC__
std::string ToString(RenderModes renderMode);
#endif



//------------------------------------------------------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------------------------------------------------------
struct RayPickResult
{
	float3 rayOrigin;
	uint32_t instIx;

	float3 rayDir;
	uint32_t primIx;

	float tmax;
};



struct LaunchParams
{
	// Other
	float3 cameraPos;
	int32_t resX;

	float3 cameraSide;
	int32_t resY;

	float3 cameraUp;
	float cameraFov;

	float3 cameraForward;
	uint32_t sampleCount;

	OptixTraversableHandle sceneRoot;
	float4* accumulator;

	// ray pick
	int2 rayPickPixel;
	RayPickResult* rayPickResult;

	// SPT
	float4* pathStates;
	uint4* hitData;

	// Shadow
	float4* shadowRays;

	// optix
	RayGenModes rayGenMode;

	// settings
	int multiSample;
	int maxDepth;
	float epsilon;
	float aoDist;
	float zDepthMax;
};



struct Counters
{
	int32_t extendRays = 0;
	int32_t shadowRays = 0;
};



//------------------------------------------------------------------------------------------------------------------------------
// Geometry
//------------------------------------------------------------------------------------------------------------------------------
struct alignas(8) PackedTriangle
{
	// vertex 0
	float v0x;
	float v0y;
	float v0z;
	half uv0x;
	half uv0y;
	half N0x;
	half N0y;
	half N0z;
	half Nx;

	// vertex 1
	float v1x;
	float v1y;
	float v1z;
	half uv1x;
	half uv1y;
	half N1x;
	half N1y;
	half N1z;
	half Ny;

	// vertex 2
	float v2x;
	float v2y;
	float v2z;
	half uv2x;
	half uv2y;
	half N2x;
	half N2y;
	half N2z;
	half Nz;
};



struct alignas(16) LightTriangle
{
	float3 V0;
	int32_t instIx;

	float3 V1;
	int32_t triIx;

	float3 V2;
	float area;

	float3 N;
	float energy;

	float3 radiance;
	float sumEnergy;
};



struct CudaMeshData
{
	PackedTriangle* triangles;
	uint32_t* materialIndices;
};



//------------------------------------------------------------------------------------------------------------------------------
// Material
//------------------------------------------------------------------------------------------------------------------------------
struct alignas(16) CudaMatarial
{
	float3 diffuse;
	uint32_t textures;

	float3 emissive;
	int pad0;

	cudaTextureObject_t diffuseMap;
	cudaTextureObject_t normalMap;
};




//------------------------------------------------------------------------------------------------------------------------------
// Sky
//------------------------------------------------------------------------------------------------------------------------------
struct SkyData
{
	float3 sunDir;
	float sunSize;

	float3 sunColor;
	int enableSun;

	float3 groundAlbedo;
	float turbidity;

	float3 skyTint;
	float dummy;

	float3 sunTint;
	float dummy2;
};



struct SkyState
{
	float configs[3][9];
	float radiances[3];
	float turbidity;
	float solarRadius;
	float albedo;
	float elevation;
};
