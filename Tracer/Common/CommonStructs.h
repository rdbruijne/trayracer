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
	//RayType_Shadow,

	RayType_Count
};



enum RayGenModes : uint32_t
{
	RayGen_Primary,
	RayGen_Secondary
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



struct alignas(16) RayPickResult
{
	float3 rayOrigin;
	uint32_t instIx;

	float3 rayDir;
	uint32_t primIx;

	float tmax;
	float pad[3];
};



struct alignas(16) LaunchParams
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

	// settings
	int rayGenMode;
	int multiSample;
	int pad[2];

	int maxDepth;
	float epsilon;
	float aoDist;
	float zDepthMax;
};



struct Counters
{
	int32_t extendRays = 0;
};



struct alignas(16) CudaMeshData
{
	float3* vertices;
	float3* normals;

	float2* texcoords;
	uint3* indices;

	uint32_t* matIndices;
	uint32_t objectID;
	uint32_t pad;
};



struct CudaMatarial
{
	float3 diffuse;
	uint32_t textures;

	float3 emissive;
	int pad;

	cudaTextureObject_t diffuseMap;
	int64_t pad2;
};
