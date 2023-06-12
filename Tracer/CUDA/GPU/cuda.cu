// kernel includes
#include "CudaLights.h"
#include "CudaMaterial.h"

#include "CudaUtilityKernels.h"

// Shade kernels
#include "Shade/AmbientOcclusion.h"
#include "Shade/AmbientOcclusionShading.h"
#include "Shade/Bitangent.h"
#include "Shade/GeometricNormal.h"
#include "Shade/MaterialID.h"
#include "Shade/MaterialProperty.h"
#include "Shade/ObjectID.h"
#include "Shade/PathTracing.h"
#include "Shade/ShadingNormal.h"
#include "Shade/Tangent.h"
#include "Shade/TextureCoordinate.h"
#include "Shade/Wireframe.h"
#include "Shade/ZDepth.h"

__host__ void Shade(RenderModes renderMode, DECLARE_KERNEL_PARAMS)
{
	const uint32_t threadsPerBlock = 128;
	const uint32_t blockCount = DivRoundUp(pathCount, threadsPerBlock);

	switch(renderMode)
	{
	case RenderModes::AmbientOcclusion:
		AmbientOcclusionKernel<<<blockCount, threadsPerBlock>>>(PASS_KERNEL_PARAMS);
		break;

	case RenderModes::AmbientOcclusionShading:
		AmbientOcclusionShadingKernel<<<blockCount, threadsPerBlock>>>(PASS_KERNEL_PARAMS);
		break;

	case RenderModes::Bitangent:
		BitangentKernel<<<blockCount, threadsPerBlock>>>(PASS_KERNEL_PARAMS);
		break;

	case RenderModes::GeometricNormal:
		GeometricNormalKernel<<<blockCount, threadsPerBlock>>>(PASS_KERNEL_PARAMS);
		break;

	case RenderModes::MaterialID:
		MaterialIDKernel<<<blockCount, threadsPerBlock>>>(PASS_KERNEL_PARAMS);
		break;

	case RenderModes::MaterialProperty:
		MaterialPropertyKernel<<<blockCount, threadsPerBlock>>>(PASS_KERNEL_PARAMS);
		break;

	case RenderModes::ObjectID:
		ObjectIDKernel<<<blockCount, threadsPerBlock>>>(PASS_KERNEL_PARAMS);
		break;

	case RenderModes::PathTracing:
		PathTracingKernel<<<blockCount, threadsPerBlock>>>(PASS_KERNEL_PARAMS);
		break;

	case RenderModes::ShadingNormal:
		ShadingNormalKernel<<<blockCount, threadsPerBlock>>>(PASS_KERNEL_PARAMS);
		break;

	case RenderModes::Tangent:
		TangentKernel<<<blockCount, threadsPerBlock>>>(PASS_KERNEL_PARAMS);
		break;

	case RenderModes::TextureCoordinate:
		TextureCoordinateKernel<<<blockCount, threadsPerBlock>>>(PASS_KERNEL_PARAMS);
		break;

	case RenderModes::Wireframe:
		WireframeKernel<<<blockCount, threadsPerBlock>>>(PASS_KERNEL_PARAMS);
		break;

	case RenderModes::ZDepth:
		ZDepthKernel<<<blockCount, threadsPerBlock>>>(PASS_KERNEL_PARAMS);
		break;

	default:
		break;
	}
}
