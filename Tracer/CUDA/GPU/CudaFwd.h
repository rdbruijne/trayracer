#pragma once

// Project
#include "Common/CommonStructs.h"

// C++
#include <stdint.h>

__host__ void SetCudaCounters(const RayCounters* data);
__host__ void SetCudaLaunchParams(const LaunchParams* data);

// geometry
__host__ void SetCudaMeshData(const CudaMeshData* data);

// materials
__host__ void SetCudaMatarialData(const CudaMatarial* data);
__host__ void SetCudaMatarialOffsets(const uint32_t* data);

// instances
__host__ void SetCudaInvTransforms(const float4* data);
__host__ void SetCudaModelIndices(const uint32_t* data);

// lights
__host__ void SetCudaLightCount(int32_t count);
__host__ void SetCudaLightEnergy(float energy);
__host__ void SetCudaLights(const LightTriangle* data);

// sky
__host__ void SetCudaSkyData(const SkyData* data);

// rendering
__host__ void InitCudaCounters();
__host__ void FinalizeFrame(float4* __restrict__ accumulator, float4* __restrict__ colors, int resX, int resY, uint32_t sampleCount);
__host__ void Shade(RenderModes renderMode, DECLARE_KERNEL_PARAMS);
