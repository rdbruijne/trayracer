#include "DeviceRenderer.h"

// Project
#include "CUDA/CudaDevice.h"
#include "Optix/OptixRenderer.h"
#include "Resources/CameraNode.h"

// GPU
#include "CUDA/GPU/CudaFwd.h"


namespace Tracer
{
	DeviceRenderer::DeviceRenderer(std::shared_ptr<CudaDevice> device) :
		mDevice(device)
	{
		// create Optix renderer
		mOptixRenderer = std::make_unique<OptixRenderer>(mDevice->Context());

		// allocate launch params
		mLaunchParamsBuffer.Alloc(sizeof(LaunchParams));
	}



	void DeviceRenderer::SetCamera(CameraNode& camNode)
	{
		const float3x4 T = camNode.Transform();

		mLaunchParams.cameraPos            = make_float3(T.tx, T.ty, T.tz);
		mLaunchParams.cameraForward        = T.z;
		mLaunchParams.cameraSide           = T.x;
		mLaunchParams.cameraUp             = T.y;

		mLaunchParams.cameraAperture       = camNode.Aperture();
		mLaunchParams.cameraDistortion     = camNode.Distortion();
		mLaunchParams.cameraFocalDist      = camNode.FocalDist();
		mLaunchParams.cameraFov            = camNode.Fov();
		mLaunchParams.cameraBokehSideCount = camNode.BokehSideCount();
		mLaunchParams.cameraBokehRotation  = camNode.BokehRotation();
	}



	void DeviceRenderer::Resize(const int2& resolution)
	{
		assert(Device()->IsCurrent());

		if(mResolution != resolution)
		{
			// resize buffers
			mAccumulator.Resize(sizeof(float4) * resolution.x * resolution.y);
			mAlbedoBuffer.Resize(sizeof(float4) * resolution.x * resolution.y);
			mNormalBuffer.Resize(sizeof(float4) * resolution.x * resolution.y);
			mColorBuffer.Resize(sizeof(float4) * resolution.x * resolution.y);

			// reset sample count
			Reset();
		}

		// update launch params
		mLaunchParams.resX = resolution.x;
		mLaunchParams.resY = resolution.y;
		mLaunchParams.accumulator = mAccumulator.Ptr<float4>();

		// store the resolution
		mResolution = resolution;
	}



	RayPickResult DeviceRenderer::PickRay(const int2& pixelIndex)
	{
		// allocate result buffer
		CudaBuffer resultBuffer;
		resultBuffer.Alloc(sizeof(RayPickResult));

		// set ray pick specific launch param options
		mLaunchParams.rayGenMode    = RayGenModes::RayPick;
		mLaunchParams.rayPickPixel  = pixelIndex;
		mLaunchParams.rayPickResult = resultBuffer.Ptr<RayPickResult>();

		// upload launch params
		mLaunchParamsBuffer.Upload(&mLaunchParams);

		// launch the kernel
		Optix()->TraceRays(mDevice->Stream(), mLaunchParamsBuffer, 1, 1, 1);

		// read the raypick result
		RayPickResult result;
		resultBuffer.Download(&result);

		return result;
	}



	void DeviceRenderer::RenderFrame(const KernelSettings& kernelSettings, const RenderModes& renderMode, uint32_t renderFlags)
	{
		assert(Device()->IsCurrent());

		// set the render mode
		mLaunchParams.renderMode  = renderMode;
		mLaunchParams.renderFlags = renderFlags;

		// set render settings
		mLaunchParams.kernelSettings = kernelSettings;

		// set device specific launch param options
		mLaunchParams.sceneRoot   = mOptixRenderer->SceneRoot();
		mLaunchParams.accumulator = mAccumulator.Ptr<float4>();

		// resize the buffer
		Resize(make_int2(mLaunchParams.resX, mLaunchParams.resY));

		PreRender();

		// loop
		uint32_t pathCount = static_cast<uint32_t>(mLaunchParams.resX) * mLaunchParams.resY * mLaunchParams.kernelSettings.multiSample;
		mRenderTimeEvent.Start(mDevice->Stream());
		int pathLength = 0;
		do
		{
			RenderBounce(static_cast<uint32_t>(pathLength), pathCount);
		} while(++pathLength < mLaunchParams.kernelSettings.maxDepth && pathCount > 0);
		mRenderTimeEvent.Stop(mDevice->Stream());
		cudaStreamSynchronize(mDevice->Stream());

		// finalize the frame
		mLaunchParams.sampleCount += mLaunchParams.kernelSettings.multiSample;
		FinalizeFrame(mAccumulator.Ptr<float4>(), mColorBuffer.Ptr<float4>(), mLaunchParams.resX, mLaunchParams.resY, mLaunchParams.sampleCount);

		PostRender();
	}



	void DeviceRenderer::PreRender()
	{
		assert(Device()->IsCurrent());

		// prepare SPT buffers
		const uint32_t stride = mLaunchParams.resX * mLaunchParams.resY * mLaunchParams.kernelSettings.multiSample;

		if(mPathStates.Size() != sizeof(float4) * stride * 3)
		{
			mPathStates.Resize(sizeof(float4) * stride * 3);
			mLaunchParams.pathStates = mPathStates.Ptr<float4>();
		}

		if(mHitData.Size() != sizeof(uint4) * stride)
		{
			mHitData.Resize(sizeof(uint4) * stride);
			mLaunchParams.hitData = mHitData.Ptr<uint4>();
		}

		if(mShadowRayData.Size() != sizeof(float4) * stride * 3)
		{
			mShadowRayData.Resize(sizeof(float4) * stride * 3);
			mLaunchParams.shadowRays = mShadowRayData.Ptr<float4>();
		}

		// create counters
		if(mCountersBuffer.Size() == 0)
		{
			RayCounters counters = {};
			mCountersBuffer.Upload(&counters, 1, true);
			SetCudaCounters(mCountersBuffer.Ptr<RayCounters>());
		}

		// update launch params
		mLaunchParams.rayGenMode = RayGenModes::Primary;
		mLaunchParamsBuffer.Upload(&mLaunchParams);
		SetCudaLaunchParams(mLaunchParamsBuffer.Ptr<LaunchParams>());

		// reset stats
		mRenderStats = {};
	}



	void DeviceRenderer::PostRender()
	{
		assert(Device()->IsCurrent());

		// update timings
		mRenderStats.primaryPathTimeMs = mTraceTimeEvents[0].Elapsed();
		mRenderStats.secondaryPathTimeMs = mLaunchParams.kernelSettings.maxDepth > 1 ? mTraceTimeEvents[1].Elapsed() : 0;
		for(int i = 2; i < mLaunchParams.kernelSettings.maxDepth; i++)
			mRenderStats.deepPathTimeMs += mTraceTimeEvents[static_cast<size_t>(i)].Elapsed();
		for(int i = 0; i < mLaunchParams.kernelSettings.maxDepth; i++)
			mRenderStats.shadeTimeMs += mShadeTimeEvents[static_cast<size_t>(i)].Elapsed();
		for(int i = 0; i < mLaunchParams.kernelSettings.maxDepth; i++)
			mRenderStats.shadowTimeMs += mShadowTimeEvents[static_cast<size_t>(i)].Elapsed();
		mRenderStats.renderTimeMs = mRenderTimeEvent.Elapsed();
	}



	void DeviceRenderer::RenderBounce(uint32_t pathLength, uint32_t& pathCount)
	{
		assert(Device()->IsCurrent());

		if(pathCount == 0)
			return;

		// prepare counters
		mTraceTimeEvents[static_cast<size_t>(pathLength)].Start(mDevice->Stream());
		InitCudaCounters();

		// trace rays
		if(pathLength == 0)
		{
			// primary
			mRenderStats.primaryPathCount = pathCount;
			mOptixRenderer->TraceRays(mDevice->Stream(), mLaunchParamsBuffer,
				static_cast<unsigned int>(mLaunchParams.resX),
				static_cast<unsigned int>(mLaunchParams.resY),
				static_cast<unsigned int>(mLaunchParams.kernelSettings.multiSample));
		}
		else if(pathCount > 0)
		{
			// bounce
			mLaunchParams.rayGenMode = RayGenModes::Secondary;
			mLaunchParamsBuffer.Upload(&mLaunchParams);
			if(pathLength == 1)
				mRenderStats.secondaryPathCount = pathCount;
			else
				mRenderStats.deepPathCount += pathCount;
			mOptixRenderer->TraceRays(mDevice->Stream(), mLaunchParamsBuffer, pathCount, 1, 1);
		}

		// update counters
		mRenderStats.pathCount += pathCount;
		mTraceTimeEvents[static_cast<size_t>(pathLength)].Stop(mDevice->Stream());

		// shade
		const uint32_t stride = mLaunchParams.resX * mLaunchParams.resY * mLaunchParams.kernelSettings.multiSample;
		mShadeTimeEvents[static_cast<size_t>(pathLength)].Start(mDevice->Stream());
		Shade(mLaunchParams.renderMode,		// RenderModes renderMode
			pathCount,						// uint32_t pathCount
			mAccumulator.Ptr<float4>(),		// float4* accumulator
			mAlbedoBuffer.Ptr<float4>(),	// float4* albedo
			mNormalBuffer.Ptr<float4>(),	// float4* normals
			mPathStates.Ptr<float4>(),		// float4* pathStates
			mHitData.Ptr<uint4>(),			// uint4* hitData
			mShadowRayData.Ptr<float4>(),	// float4* shadowRays
			mLaunchParams.resX,				// int resX
			mLaunchParams.resY,				// int resY
			stride,							// uint32_t stride
			pathLength,						// uint32_t pathLength
			mLaunchParams.renderFlags);		// uint32_t renderFlags
		mShadeTimeEvents[static_cast<size_t>(pathLength)].Stop(mDevice->Stream());

		// update counters
		RayCounters counters = {};
		mCountersBuffer.Download(&counters, 1);
		pathCount = counters.extendRays;

		// shadow rays
		if(counters.shadowRays > 0)
		{
			// fire shadow rays
			mShadowTimeEvents[pathLength].Start(mDevice->Stream());
			mLaunchParams.rayGenMode = RayGenModes::Shadow;
			mLaunchParamsBuffer.Upload(&mLaunchParams);
			mOptixRenderer->TraceRays(mDevice->Stream(), mLaunchParamsBuffer, counters.shadowRays, 1, 1);
			mShadowTimeEvents[pathLength].Stop(mDevice->Stream());

			// update stats
			mRenderStats.shadowRayCount += counters.shadowRays;
			mRenderStats.pathCount += counters.shadowRays;
			counters.shadowRays = 0;
		}
	}
}
