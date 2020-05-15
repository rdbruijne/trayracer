#pragma once

// Optix
#include "optix7/optix.h"

#ifndef __CUDACC__
// Magic Enum
#pragma warning(push)
#pragma warning(disable: 5027)
#include "magic_enum/magic_enum.hpp"
#pragma warning(pop)
#endif

// CUDA
#include <cuda_runtime.h>

// C++
#include <stdint.h>

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
};



enum TexturesInMaterial
{
	Texture_DiffuseMap = 0x1
};



enum class RenderModes : uint32_t
{
	AmbientOcclusion,
	AmbientOcclusionShading,
	DiffuseFilter,
	DirectLight,
	GeometricNormal,
	MaterialID,
	ObjectID,
	PathTracing,
	ShadingNormal,
	TextureCoordinate,
	Wireframe,
	ZDepth
};
#ifndef __CUDACC__
std::string ToString(RenderModes renderMode);
#endif



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
	int32_t sampleCount;

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
	float3 skyColor;
};



struct Counters
{
	int32_t extendRays = 0;
	int32_t shadowRays = 0;
};



struct alignas(16) PackedTriangle
{
	float2 uv0;
	float2 uv1;

	float2 uv2;
	uint32_t matIx;
	uint32_t pad0;

	float3 N0;
	float Nx;

	float3 N1;
	float Ny;

	float3 N2;
	float Nz;

	float3 tangent;
	float pad1;

	float3 bitangent;
	float pad2;
};



struct alignas(16) LightTriangle
{
	float3 V0; int32_t instIx;
	float3 V1; int32_t triIx;
	float3 V2; float area;
	float3 N;  float energy;
	float3 radiance; float dummy;
};



struct CudaMeshData
{
	PackedTriangle* triangles;
};



struct alignas(16) CudaMatarial
{
	float3 diffuse;
	uint32_t textures;

	float3 emissive;
	int pad0;

	cudaTextureObject_t diffuseMap;
	int64_t pad1;
};
