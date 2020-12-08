#pragma once

// Project
#include "Common/CommonStructs.h"
#include "CUDA/CudaBuffer.h"
#include "Utility/LinearMath.h"

namespace Tracer
{
	class Sky
	{
	public:
		// dirty state
		inline bool IsDirty() { return mIsDirty; }
		inline void MarkDirty() { mIsDirty = true; }
		inline void MarkClean() { mIsDirty = false; }

		// Sky
		bool Enabled() const { return mEnabled; }
		void SetEnabled(bool enabled);

		// sun
		bool DrawSun() const { return mDrawSun; }
		void SetDrawSun(bool draw);

		const float3& SunDir() const { return mSunDir; }
		void SetSunDir(const float3& dir);

		// angular diameter is in arc minutes
		float SunAngularDiameter() const { return mSunAngularDiameter; }
		void SetSunAngularDiameter(float size);

		float SunIntensity() const { return mSunIntensity; }
		void SetSunIntensity(float intensity);

		// ground
		float Turbidity() const { return mTurbidity; }
		void SetTurbidity(float turbidity);

		// build
		void Build();

		// build info
		const SkyData& CudaData() const { return mCudaData; }

	private:
		// dirty state
		bool mIsDirty = true;

		// sky
		bool mEnabled = true;

		// sun
		bool mDrawSun = true;
		float3 mSunDir = make_float3(0, 1, 0);
		float mSunAngularDiameter = 32.f;
		float mSunIntensity = 1000.f;

		// ground
		float mTurbidity = 4.f;

		// CUDA data
		SkyData mCudaData = {};
	};
}
