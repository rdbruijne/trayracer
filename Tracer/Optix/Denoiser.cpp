#include "Denoiser.h"

// Project
#include "Optix/OptixError.h"

// Optix
#pragma warning(push)
#pragma warning(disable: 4061)
#include "optix7/optix_stubs.h"
#pragma warning(pop)

namespace Tracer
{
	Denoiser::Denoiser(OptixDeviceContext optixContext)
	{
		OptixDenoiserOptions denoiserOptions;
		denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;

		mHdrIntensity.Alloc(sizeof(float));

		OPTIX_CHECK(optixDenoiserCreate(optixContext, &denoiserOptions, &mOptixDenoiser));
		OPTIX_CHECK(optixDenoiserSetModel(mOptixDenoiser, OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0));
	}



	Denoiser::~Denoiser()
	{
		OPTIX_CHECK(optixDenoiserDestroy(mOptixDenoiser));
	}



	void Denoiser::Resize(const int2& resolution)
	{
		mDenoised.Resize(sizeof(float4) * resolution.x * resolution.y);

		OptixDenoiserSizes denoiserReturnSizes;
		OPTIX_CHECK(optixDenoiserComputeMemoryResources(mOptixDenoiser, resolution.x, resolution.y, &denoiserReturnSizes));
		mScratch.Resize(denoiserReturnSizes.withoutOverlapScratchSizeInBytes);
		mState.Resize(denoiserReturnSizes.stateSizeInBytes);
		OPTIX_CHECK(optixDenoiserSetup(mOptixDenoiser, 0, resolution.x, resolution.y,
									   mState.DevicePtr(), mState.Size(),
									   mScratch.DevicePtr(), mScratch.Size()));
	}



	void Denoiser::DenoiseFrame(CUstream stream, const int2& resolution, uint32_t sampleCount,
								const CudaBuffer& colorBuffer, const CudaBuffer& albedoBuffer, const CudaBuffer& normalBuffer)
	{
		// input
		OptixImage2D inputLayers[3];

		// rgb input
		inputLayers[0].data               = colorBuffer.DevicePtr();
		inputLayers[0].width              = resolution.x;
		inputLayers[0].height             = resolution.y;
		inputLayers[0].rowStrideInBytes   = resolution.x * sizeof(float4);
		inputLayers[0].pixelStrideInBytes = sizeof(float4);
		inputLayers[0].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

		// albedo input
		inputLayers[1].data               = albedoBuffer.DevicePtr();
		inputLayers[1].width              = resolution.x;
		inputLayers[1].height             = resolution.y;
		inputLayers[1].rowStrideInBytes   = resolution.x * sizeof(float4);
		inputLayers[1].pixelStrideInBytes = sizeof(float4);
		inputLayers[1].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

		// normal input
		inputLayers[2].data               = normalBuffer.DevicePtr();
		inputLayers[2].width              = resolution.x;
		inputLayers[2].height             = resolution.y;
		inputLayers[2].rowStrideInBytes   = resolution.x * sizeof(float4);
		inputLayers[2].pixelStrideInBytes = sizeof(float4);
		inputLayers[2].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

		// output
		OptixImage2D outputLayer;
		outputLayer.data                  = mDenoised.DevicePtr();
		outputLayer.width                 = resolution.x;
		outputLayer.height                = resolution.y;
		outputLayer.rowStrideInBytes      = resolution.x * sizeof(float4);
		outputLayer.pixelStrideInBytes    = sizeof(float4);
		outputLayer.format                = OPTIX_PIXEL_FORMAT_FLOAT4;

		// calculate intensity
		OPTIX_CHECK(optixDenoiserComputeIntensity(mOptixDenoiser, stream, &inputLayers[0], mHdrIntensity.DevicePtr(),
													mScratch.DevicePtr(), mScratch.Size()));

		// denoise
		OptixDenoiserParams denoiserParams;
		denoiserParams.denoiseAlpha = 1;
		denoiserParams.hdrIntensity = mHdrIntensity.DevicePtr();
		denoiserParams.blendFactor  = 1.f / (sampleCount + 1);

		OPTIX_CHECK(optixDenoiserInvoke(mOptixDenoiser, stream, &denoiserParams, mState.DevicePtr(), mState.Size(),
										inputLayers, 3, 0, 0, &outputLayer,
										mScratch.DevicePtr(), mScratch.Size()));

		mSampleCount = sampleCount;
	}
}
