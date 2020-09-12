#include "Sky.h"

// HosekSky
#include "HosekSky/ArHosekSkyModel.h"

namespace Tracer
{
	Sky::~Sky()
	{
		FreeStates();
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



	void Sky::SetSunSize(float size)
	{
		mSunSize = size;
		MarkDirty();
	}



	void Sky::SetSunColor(const float3& color)
	{
		mSunColor = color;
		MarkDirty();
	}



	void Sky::SetSkyTint(const float3& tint)
	{
		mSkyTint = tint;
		MarkDirty();
	}



	void Sky::SetSunTint(const float3& tint)
	{
		mSunTint = tint;
		MarkDirty();
	}



	void Sky::SetTurbidity(float turbidity)
	{
		mTurbidity = clamp(turbidity, 1.f, 10.f);
		MarkDirty();
	}



	void Sky::SetGroundAlbedo(const float3& groundAlbedo)
	{
		mGroundAlbedo = groundAlbedo;
		MarkDirty();
	}



	void Sky::Build()
	{
		const float thetaS = acosf(dot(mSunDir, make_float3(0, 1, 0)));
		const float elevation = static_cast<float>(M_PI_2) - thetaS;

		// clean up old data
		FreeStates();

		// init states
		mStateX = arhosek_xyz_skymodelstate_alloc_init(mTurbidity, mGroundAlbedo.x, elevation);
		mStateY = arhosek_xyz_skymodelstate_alloc_init(mTurbidity, mGroundAlbedo.y, elevation);
		mStateZ = arhosek_xyz_skymodelstate_alloc_init(mTurbidity, mGroundAlbedo.z, elevation);

		// fill data
		mCudaData.sunDir = mSunDir;
		mCudaData.sunSize = cosf(std::max(mSunSize, .01f) * DegToRad);

		mCudaData.sunColor = mSunColor;
		mCudaData.enableSun = mDrawSun ? 1 : 0;

		mCudaData.groundAlbedo = mGroundAlbedo;
		mCudaData.turbidity = mTurbidity;

		mCudaData.skyTint = mSkyTint;
		mCudaData.sunTint = mSunTint;

		// fill states
		SetState(*mStateX, mCudaStateX);
		SetState(*mStateY, mCudaStateY);
		SetState(*mStateZ, mCudaStateZ);

		MarkClean();
	}



	void Sky::FreeStates()
	{
		if(mStateX)
		{
			arhosekskymodelstate_free(mStateX);
			mStateX = nullptr;
		}

		if(mStateY)
		{
			arhosekskymodelstate_free(mStateY);
			mStateY = nullptr;
		}

		if(mStateZ)
		{
			arhosekskymodelstate_free(mStateZ);
			mStateZ = nullptr;
		}
	}



	void Sky::SetState(ArHosekSkyModelState& cpu, SkyState& cuda) const
	{
		cuda.turbidity   = static_cast<float>(cpu.turbidity);
		cuda.solarRadius = static_cast<float>(cpu.solar_radius);
		cuda.albedo      = static_cast<float>(cpu.albedo);
		cuda.elevation   = static_cast<float>(cpu.elevation);

		for(int i = 0; i < 3; i++)
		{
			for(int j = 0; j < 9; j++)
				cuda.configs[i][j] = static_cast<float>(cpu.configs[i][j]);
			cuda.radiances[i] = static_cast<float>(cpu.radiances[i]);
		}
	}
}
