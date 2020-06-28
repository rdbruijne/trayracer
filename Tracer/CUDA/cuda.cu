#include "CudaUtilityKernels.h"
#include "CudaUtility.h"

// Shade kernels
#include "Shade/AmbientOcclusion.h"
#include "Shade/AmbientOcclusionShading.h"
#include "Shade/DiffuseFilter.h"
#include "Shade/DirectLight.h"
#include "Shade/GeometricNormal.h"
#include "Shade/MaterialID.h"
#include "Shade/ObjectID.h"
#include "Shade/PathTracing.h"
#include "Shade/ShadingNormal.h"
#include "Shade/TextureCoordinate.h"
#include "Shade/Wireframe.h"
#include "Shade/ZDepth.h"



__host__ void Shade(RenderModes renderMode, uint32_t pathCount, float4* accumulator, float4* pathStates, uint4* hitData, float4* shadowRays, int2 resolution, uint32_t stride, uint32_t pathLength)
{
	const uint32_t threadsPerBlock = 128;
	const uint32_t blockCount = DivRoundUp(pathCount, threadsPerBlock);
	switch(renderMode)
	{
	case RenderModes::AmbientOcclusion:
		AmbientOcclusionKernel<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, shadowRays, resolution, stride, pathLength);
		break;

	case RenderModes::AmbientOcclusionShading:
		AmbientOcclusionShadingKernel<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, shadowRays, resolution, stride, pathLength);
		break;

	case RenderModes::DiffuseFilter:
		DiffuseFilterKernel<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, shadowRays, resolution, stride, pathLength);
		break;

	case RenderModes::DirectLight:
		DirectLightKernel<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, shadowRays, resolution, stride, pathLength);
		break;

	case RenderModes::GeometricNormal:
		GeometricNormalKernel<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, shadowRays, resolution, stride, pathLength);
		break;

	case RenderModes::MaterialID:
		MaterialIDKernel<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, shadowRays, resolution, stride, pathLength);
		break;

	case RenderModes::ObjectID:
		ObjectIDKernel<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, shadowRays, resolution, stride, pathLength);
		break;

	case RenderModes::PathTracing:
		PathTracingKernel<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, shadowRays, resolution, stride, pathLength);
		break;

	case RenderModes::ShadingNormal:
		ShadingNormalKernel<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, shadowRays, resolution, stride, pathLength);
		break;

	case RenderModes::TextureCoordinate:
		TextureCoordinateKernel<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, shadowRays, resolution, stride, pathLength);
		break;

	case RenderModes::Wireframe:
		WireframeKernel<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, shadowRays, resolution, stride, pathLength);
		break;

	case RenderModes::ZDepth:
		ZDepthKernel<<<blockCount, threadsPerBlock>>>(pathCount, accumulator, pathStates, hitData, shadowRays, resolution, stride, pathLength);
		break;

	default:
		break;
	}
}
