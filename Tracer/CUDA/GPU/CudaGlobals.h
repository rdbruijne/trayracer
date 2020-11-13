#pragma once

#include "Common/CommonStructs.h"

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
