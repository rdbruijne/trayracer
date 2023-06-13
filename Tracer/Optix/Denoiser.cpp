#include "Denoiser.h"

// Project
#include "Optix/OptixError.h"

// Optix
#include "optix7/optix_stubs.h"

namespace Tracer
{
	Denoiser::Denoiser(OptixDeviceContext optixContext)
	{
		OptixDenoiserOptions denoiserOptions;
		denoiserOptions.guideAlbedo = 1;
		denoiserOptions.guideNormal = 1;

		mHdrIntensity.Alloc(sizeof(float));

		OPTIX_CHECK(optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_HDR, &denoiserOptions, &mOptixDenoiser));
	}



	Denoiser::~Denoiser()
	{
		OPTIX_CHECK(optixDenoiserDestroy(mOptixDenoiser));
	}



	void Denoiser::Resize(const int2& resolution)
	{
		mDenoised.Resize(sizeof(float4) * resolution.x * resolution.y);

		OptixDenoiserSizes denoiserReturnSizes;
		OPTIX_CHECK(optixDenoiserComputeMemoryResources(mOptixDenoiser, static_cast<unsigned int>(resolution.x), static_cast<unsigned int>(resolution.y), &denoiserReturnSizes));
		mScratch.Resize(denoiserReturnSizes.withoutOverlapScratchSizeInBytes);
		mState.Resize(denoiserReturnSizes.stateSizeInBytes);
		OPTIX_CHECK(optixDenoiserSetup(mOptixDenoiser, 0,
									   static_cast<unsigned int>(resolution.x), static_cast<unsigned int>(resolution.y),
									   mState.DevicePtr(), mState.Size(),
									   mScratch.DevicePtr(), mScratch.Size()));
	}



	void Denoiser::DenoiseFrame(CUstream stream, const int2& resolution, uint32_t sampleCount,
								const CudaBuffer& colorBuffer, const CudaBuffer& albedoBuffer, const CudaBuffer& normalBuffer)
	{
		const uint32_t resX = static_cast<uint32_t>(resolution.x);
		const uint32_t resY = static_cast<uint32_t>(resolution.y);
		const uint32_t stride = resX * sizeof(float4);

		// guide layer
		OptixDenoiserGuideLayer guideLayer;

		// albedo/bsdf image
		guideLayer.albedo.data               = albedoBuffer.DevicePtr();
		guideLayer.albedo.width              = resX;
		guideLayer.albedo.height             = resY;
		guideLayer.albedo.rowStrideInBytes   = stride;
		guideLayer.albedo.pixelStrideInBytes = sizeof(float4);
		guideLayer.albedo.format             = OPTIX_PIXEL_FORMAT_FLOAT4;

		// normal vector image (2d or 3d pixel format)
		guideLayer.normal.data               = normalBuffer.DevicePtr();
		guideLayer.normal.width              = resX;
		guideLayer.normal.height             = resY;
		guideLayer.normal.rowStrideInBytes   = stride;
		guideLayer.normal.pixelStrideInBytes = sizeof(float4);
		guideLayer.normal.format             = OPTIX_PIXEL_FORMAT_FLOAT4;
		
		// 2d flow image, pixel flow from previous to current frame for each pixel
		//OptixImage2D  guideLayer.flow;
		//OptixImage2D  guideLayer.previousOutputInternalGuideLayer;
		//OptixImage2D  guideLayer.outputInternalGuideLayer;

		// denoiser layers
		OptixDenoiserLayer layer;

		// input image (beauty or AOV)
		layer.input.data               = colorBuffer.DevicePtr();
		layer.input.width              = resX;
		layer.input.height             = resY;
		layer.input.rowStrideInBytes   = stride;
		layer.input.pixelStrideInBytes = sizeof(float4);
		layer.input.format             = OPTIX_PIXEL_FORMAT_FLOAT4;

	    // denoised output image from previous frame if temporal model kind selected
	    //OptixImage2D  layers.previousOutput;

	    // denoised output for given input
		layer.output.data                  = mDenoised.DevicePtr();
		layer.output.width                 = resX;
		layer.output.height                = resY;
		layer.output.rowStrideInBytes      = stride;
		layer.output.pixelStrideInBytes    = sizeof(float4);
		layer.output.format                = OPTIX_PIXEL_FORMAT_FLOAT4;

		// calculate intensity
		OPTIX_CHECK(optixDenoiserComputeIntensity(mOptixDenoiser, stream, &layer.input, mHdrIntensity.DevicePtr(),
													mScratch.DevicePtr(), mScratch.Size()));

		// denoise
		OptixDenoiserParams denoiserParams;
		denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
		denoiserParams.hdrIntensity = mHdrIntensity.DevicePtr();
		denoiserParams.blendFactor  = 1.f / static_cast<float>(sampleCount + 1);

		OPTIX_CHECK(optixDenoiserInvoke(mOptixDenoiser, stream, &denoiserParams, mState.DevicePtr(), mState.Size(),
										&guideLayer, &layer, 1, 0, 0,
										mScratch.DevicePtr(), mScratch.Size()));

		mSampleCount = sampleCount;
	}
}
