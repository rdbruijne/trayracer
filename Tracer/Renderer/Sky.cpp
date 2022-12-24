#include "Sky.h"

// Project
#include "CUDA/CudaBuffer.h"
#include "Renderer/Renderer.h"

namespace Tracer
{
	void Sky::SetEnabled(bool enabled)
	{
		mEnabled = enabled;
		MarkDirty();
	}



	void Sky::SetDrawSun(bool draw)
	{
		mDrawSun = draw;
		MarkDirty();
	}



	void Sky::SetSunDir(const float3& dir)
	{
		mSunDir = normalize(dir);
		MarkDirty();
	}



	void Sky::SetSunAngularDiameter(float size)
	{
		mSunAngularDiameter = size;
		MarkDirty();
	}



	void Sky::SetSunIntensity(float intensity)
	{
		mSunIntensity = intensity;
		MarkDirty();
	}



	void Sky::SetTurbidity(float turbidity)
	{
		mTurbidity = clamp(turbidity, 1.f, 10.f);
		MarkDirty();
	}



	void Sky::SetSelectionBias(float bias)
	{
		mSelectionBias = bias;
		MarkDirty();
	}



	float Sky::SunEnergy() const
	{
		const float sunDiameterRadians = mSunAngularDiameter * Pi / (60.f * 180.f);
		const float sunRadiusRadians   = sunDiameterRadians * .5f;
		const float sunArea            = Pi * sunRadiusRadians * sunRadiusRadians;
		const float selectionBias      = exp(mSelectionBias);

		return mEnabled ? mSunIntensity * sunArea * exp(mSelectionBias) : 0;
	}



	void Sky::Build()
	{
		// dirty check
		if(!IsDirty())
			return;

		const float sunDiameterRadians = mSunAngularDiameter * Pi / (60.f * 180.f);
		const float sunRadiusRadians   = sunDiameterRadians * .5f;
		const float sunArea            = Pi * sunRadiusRadians * sunRadiusRadians;
		const float selectionBias      = exp(mSelectionBias);

		// fill data
		mSkyData.sunDir                = mSunDir;
		mSkyData.skyEnabled            = mEnabled;
		mSkyData.drawSun               = mDrawSun ? 1.f : 0.f;
		mSkyData.sunArea               = sunArea;
		mSkyData.cosSunAngularDiameter = cosf(sunDiameterRadians);
		mSkyData.sunIntensity          = mSunIntensity;
		mSkyData.turbidity             = mTurbidity;
		mSkyData.selectionBias         = selectionBias;
		mSkyData.sunEnergy             = mEnabled ? mSunIntensity * sunArea * selectionBias : 0;

		// mark out of sync
		MarkOutOfSync();

		// mark clean
		MarkClean();
	}



	void Sky::Upload(Renderer* renderer)
	{
		// sync check
		if(!IsOutOfSync())
			return;

		mCudaData.UploadAsync(&mSkyData, 1, true);

		// mark synced
		MarkSynced();
	}
}
