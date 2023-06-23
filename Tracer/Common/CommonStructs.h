#pragma once

// Optix
#include "optix7/optix.h"

#ifndef __CUDACC__
// Magic Enum
#include "magic_enum/magic_enum.hpp"
#endif

// CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// C++
#include <stdint.h>

//------------------------------------------------------------------------------------------------------------------------------
// params
//------------------------------------------------------------------------------------------------------------------------------
#define DECLARE_KERNEL_PARAMS	uint32_t pathCount, float4* __restrict__ accumulator, float4* __restrict__ albedo, \
								float4* __restrict__ normals, float4* __restrict__ pathStates, \
								uint4* hitData, float4* shadowRays, int resX, int resY, \
								uint32_t stride, uint32_t pathLength, uint32_t renderFlags

#define PASS_KERNEL_PARAMS		pathCount, accumulator, albedo, normals, pathStates, hitData, shadowRays, resX, resY, stride, pathLength, renderFlags



//------------------------------------------------------------------------------------------------------------------------------
// PathIx & Flags packing
//------------------------------------------------------------------------------------------------------------------------------
// Max pathIx (1920x1080 = 2'073'600 pixels):
//   shift  0: 4'294'967'295
//   shift  1: 2'147'483'647
//   shift  2: 1'073'741'823
//   shift  3:   536'870'911
//   shift  4:   268'435'455
//   shift  5:   134'217'727
//   shift  6:    67'108'863
//   shift  7:    33'554'431
//   shift  8:    16'777'215
//   shift  9:     8'388'607
//   shift 10:     4'194'303

constexpr uint32_t PackedPathIxShift = 4;
constexpr uint32_t PackedFlagsMask = (1u << PackedPathIxShift) - 1;

static __forceinline__ __device__
uint32_t PathIx(uint32_t packed)
{
	return packed >> PackedPathIxShift;
}



static __forceinline__ __device__
uint32_t Flags(uint32_t packed)
{
	return packed & PackedFlagsMask;
}



static __forceinline__ __device__
uint32_t Pack(uint32_t pathIx, uint32_t flags = 0)
{
	return (pathIx << PackedPathIxShift) | (flags & PackedFlagsMask);
}



//------------------------------------------------------------------------------------------------------------------------------
// Enumerations
//------------------------------------------------------------------------------------------------------------------------------
enum class MaterialPropertyIds : uint32_t
{
	Anisotropic,
	Clearcoat,
	ClearcoatGloss,
	Diffuse,
	Emissive,
	Metallic,
	Normal,
	Roughness,
	Sheen,
	SheenTint,
	Specular,
	SpecularTint,
	Subsurface,
#ifdef __CUDACC__
	_Count
#endif
};



enum class RayGenModes : uint32_t
{
	Primary,
	Secondary,
	Shadow,
	RayPick
};



enum class RenderModes : uint32_t
{
	AmbientOcclusion,
	AmbientOcclusionShading,
	Bitangent,
	GeometricNormal,
	MaterialID,
	MaterialProperty,
	ObjectID,
	PathTracing,
	ShadingNormal,
	Tangent,
	TextureCoordinate,
	Wireframe,
	ZDepth
};



//------------------------------------------------------------------------------------------------------------------------------
// OptiX
//------------------------------------------------------------------------------------------------------------------------------
struct RayPickResult
{
	float3 rayOrigin;
	uint32_t instIx;

	float3 rayDir;
	uint32_t primIx;

	float tmax;
};



struct KernelSettings
{
	int multiSample;
	int maxDepth;
	float aoDist;
	float zDepthMax;
	float rayEpsilon;
};



struct LaunchParams
{
	// film
	uint32_t resX;
	uint32_t resY;
	uint32_t sampleCount;
	int dummy;

	float4* accumulator;

	// scene
	OptixTraversableHandle sceneRoot;

	// Camera
	float3 cameraPos;
	float cameraAperture;

	float3 cameraSide;
	float cameraDistortion;

	float3 cameraUp;
	float cameraFocalDist;

	float3 cameraForward;
	float cameraFov;

	int cameraBokehSideCount;
	float cameraBokehRotation;

	// ray pick
	int2 rayPickPixel;
	RayPickResult* rayPickResult;

	// SPT
	float4* pathStates;
	uint4* hitData;

	// Shadow
	float4* shadowRays;

	// render modes
	RayGenModes rayGenMode;
	RenderModes renderMode;
	uint32_t renderFlags;

	// render settings
	KernelSettings kernelSettings;
};



struct RayCounters
{
	uint32_t extendRays = 0;
	uint32_t shadowRays = 0;
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
struct alignas(16) CudaMaterialProperty
{
	cudaTextureObject_t textureMap;
	half r;
	half g;
	half b;
	uint8_t colorChannels;
	uint8_t useTexture;
};



struct alignas(16) CudaMatarial
{
#ifdef __CUDACC__
	CudaMaterialProperty properties[static_cast<size_t>(MaterialPropertyIds::_Count)];
#else
	CudaMaterialProperty properties[magic_enum::enum_count<MaterialPropertyIds>()];
#endif
};




//------------------------------------------------------------------------------------------------------------------------------
// Sky
//------------------------------------------------------------------------------------------------------------------------------
struct SkyData
{
	float3 sunDir;
	int skyEnabled;

	float drawSun;
	float sunArea;
	float cosSunAngularDiameter;
	float sunIntensity;

	float turbidity;
	float selectionBias;
	float sunEnergy;
	float dummy;
};
