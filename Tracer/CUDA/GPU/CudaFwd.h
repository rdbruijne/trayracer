#pragma once

// Project
#include "Common/CommonStructs.h"

// C++
#include <stdint.h>

__host__ void SetCudaCounters(Counters* data);
__host__ void SetCudaLaunchParams(LaunchParams* data);

// geometry
__host__ void SetCudaMeshData(CudaMeshData* data);

// materials
__host__ void SetCudaMatarialData(CudaMatarial* data);
__host__ void SetCudaMatarialOffsets(uint32_t* data);

// instances
__host__ void SetCudaInvTransforms(float4* data);
__host__ void SetCudaModelIndices(uint32_t* data);

// lights
__host__ void SetCudaLightCount(int32_t count);
__host__ void SetCudaLightEnergy(float energy);
__host__ void SetCudaLights(LightTriangle* data);

// sky
__host__ void SetCudaSkyData(SkyData* data);
__host__ void SetCudaSkyStateX(SkyState* data);
__host__ void SetCudaSkyStateY(SkyState* data);
__host__ void SetCudaSkyStateZ(SkyState* data);



// rendering
__host__ void InitCudaCounters();
__host__ void FinalizeFrame(float4* accumulator, float4* colors, int2 resolution, int sampleCount);
__host__ void Shade(RenderModes renderMode, DECLARE_KERNEL_PARAMS);
