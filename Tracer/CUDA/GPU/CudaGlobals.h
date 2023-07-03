#pragma once

#include "Common/CommonStructs.h"

//------------------------------------------------------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------------------------------------------------------
// rendering
//constexpr float DstMax = 1e30f;
constexpr float Epsilon = 1e-3f;



//------------------------------------------------------------------------------------------------------------------------------
// Globals
//------------------------------------------------------------------------------------------------------------------------------
static __device__ RayCounters* Counters = nullptr;

__constant__ LaunchParams* Params = nullptr;

// geometry
__constant__ CudaMeshData* MeshData = nullptr;

// materials
__constant__ CudaMatarial* MaterialData = nullptr;
__constant__ uint32_t* MaterialOffsets = nullptr;

// instances
__constant__ float4* InvInstTransforms = nullptr;
__constant__ uint32_t* ModelIndices = nullptr;

// lights
__constant__ int32_t LightCount = 0;
__constant__ float LightEnergy = 0;
__constant__ LightTriangle* Lights = nullptr;

// sky
__constant__ SkyData* Sky = nullptr;



//------------------------------------------------------------------------------------------------------------------------------
// Global setters
//------------------------------------------------------------------------------------------------------------------------------
__host__ void SetCudaCounters(const RayCounters* data)
{
	cudaMemcpyToSymbol(Counters, &data, sizeof(void*));
}



__host__ void SetCudaLaunchParams(const LaunchParams* data)
{
	cudaMemcpyToSymbol(Params, &data, sizeof(void*));
}



// geometry
__host__ void SetCudaMeshData(const CudaMeshData* data)
{
	cudaMemcpyToSymbol(MeshData, &data, sizeof(void*));
}



// materials
__host__ void SetCudaMatarialData(const CudaMatarial* data)
{
	cudaMemcpyToSymbol(MaterialData, &data, sizeof(void*));
}



__host__ void SetCudaMatarialOffsets(const uint32_t* data)
{
	cudaMemcpyToSymbol(MaterialOffsets, &data, sizeof(void*));
}



// instances
__host__ void SetCudaInvTransforms(const float4* data)
{
	cudaMemcpyToSymbol(InvInstTransforms, &data, sizeof(void*));
}



__host__ void SetCudaModelIndices(const uint32_t* data)
{
	cudaMemcpyToSymbol(ModelIndices, &data, sizeof(void*));
}



// lights
__host__ void SetCudaLightCount(int32_t count)
{
	cudaMemcpyToSymbol(LightCount, &count, sizeof(LightCount));
}



__host__ void SetCudaLightEnergy(float energy)
{
	cudaMemcpyToSymbol(LightEnergy, &energy, sizeof(LightEnergy));
}



__host__ void SetCudaLights(const LightTriangle* data)
{
	cudaMemcpyToSymbol(Lights, &data, sizeof(void*));
}



// sky
__host__ void SetCudaSkyData(const SkyData* data)
{
	cudaMemcpyToSymbol(Sky, &data, sizeof(void*));
}
