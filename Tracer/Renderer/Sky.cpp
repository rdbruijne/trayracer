#include "Sky.h"

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



	void Sky::Build()
	{
		const float sunDiameterRadians = mSunAngularDiameter * Pi / (60.f * 180.f);
		const float sunRadiusRadians   = sunDiameterRadians * .5f;

		// fill data
		mCudaData.sunDir                = mSunDir;
		mCudaData.skyEnabled            = mEnabled;
		mCudaData.drawSun               = mDrawSun ? 1.f : 0.f;
		mCudaData.sunArea               = Pi * sunRadiusRadians * sunRadiusRadians;
		mCudaData.cosSunAngularDiameter = cosf(sunDiameterRadians);
		mCudaData.sunIntensity          = mSunIntensity;
		mCudaData.turbidity             = mTurbidity;

		MarkClean();
	}
}
