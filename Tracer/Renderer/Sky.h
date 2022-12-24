#pragma once

// Project
#include "Common/CommonStructs.h"
#include "CUDA/CudaBuffer.h"
#include "Resources/Resource.h"
#include "Utility/LinearMath.h"

namespace Tracer
{
	class Renderer;
	class Sky : public Resource
	{
	public:
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

		// selection bias
		float SelectionBias() const { return mSelectionBias; }
		void SetSelectionBias(float bias);

		// for statistics
		float SunEnergy() const;

		// build
		void Build();

		// upload
		void Upload(Renderer* renderer);

		// build info
		const CudaBuffer& CudaData() const { return mCudaData; }

	private:
		// sky
		bool mEnabled = true;

		// sun
		bool mDrawSun = true;
		float3 mSunDir = make_float3(0, 1, 0);
		float mSunAngularDiameter = 32.f;
		float mSunIntensity = 1000.f;

		// ground
		float mTurbidity = 4.f;

		// selection bias
		float mSelectionBias = 1.f;

		// build data
		SkyData mSkyData = {};

		// GPU data
		CudaBuffer mCudaData = {};
	};
}
