#pragma once

// Project
#include "Common/CommonStructs.h"

// C++
#include <stdint.h>

__host__ void SetCudaCounters(Counters* data);
__host__ void SetCudaLights(LightTriangle* data);
__host__ void SetCudaLightCount(int32_t count);
__host__ void SetCudaMatarialData(CudaMatarial* data);
__host__ void SetCudaMeshData(CudaMeshData* data);
__host__ void SetCudaMatarialOffsets(uint32_t* data);
__host__ void SetCudaModelIndices(uint32_t* data);
__host__ void SetCudaLaunchParams(LaunchParams* data);

__host__ void InitCudaCounters();

__host__ void FinalizeFrame(float4* accumulator, float4* colors, int2 resolution, int sampleCount);
__host__ void Shade(RenderModes renderMode, uint32_t pathCount, float4* accumulator, float4* pathStates, uint4* hitData,
					float4* shadowRays, int2 resolution, uint32_t stride, uint32_t pathLength);
