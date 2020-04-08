#pragma once

// Project
#include "Optix/Optix7.h"

// CUDA
#include <vector_types.h>

// C++
#include <stdint.h>

enum RayTypes
{
	RayType_Surface = 0,

	RayType_Count
};



enum TexturesInMaterial
{
	Texture_DiffuseMap = 0x1
};



struct alignas(16) LaunchParams
{
	float3 cameraPos;
	int32_t resolutionX;

	float3 cameraSide;
	int32_t resolutionY;

	float3 cameraUp;
	int32_t sampleCount;

	float3 cameraForward;
	float cameraFov;

	OptixTraversableHandle sceneRoot;
	float4* colorBuffer;

	int maxDepth;
	float epsilon;
	float aoDist;
	float zDepthMaX;
};



struct alignas(16) TriangleMeshData
{
	float3* vertices;
	float3* normals;

	float3* texcoords;
	uint3* indices;

	uint32_t objectID;
	float3 diffuse;

	uint32_t textures;
	uint32_t pad;
	cudaTextureObject_t diffuseMap;

	float3 emissive;
	float pad2;
};
