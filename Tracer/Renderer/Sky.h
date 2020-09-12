#pragma once

// Project
#include "Common/CommonStructs.h"
#include "Renderer/CudaBuffer.h"
#include "Utility/LinearMath.h"

struct ArHosekSkyModelState;

namespace Tracer
{
	class Sky
	{
	public:
		~Sky();

		// dirty state
		inline bool IsDirty() { return mIsDirty; }
		inline void MarkDirty() { mIsDirty = true; }
		inline void MarkClean() { mIsDirty = false; }

		// sun
		bool DrawSun() const { return mDrawSun; }
		void SetDrawSun(bool draw);

		const float3& SunDir() const { return mSunDir; }
		void SetSunDir(const float3& dir);

		float SunSize() const { return mSunSize; }
		void SetSunSize(float size);

		const float3& SunColor() const { return mSunColor; }
		void SetSunColor(const float3& color);

		// tint
		const float3& SkyTint() const { return mSkyTint; }
		void SetSkyTint(const float3& tint);

		const float3& SunTint() const { return mSunTint; }
		void SetSunTint(const float3& tint);

		// ground
		float Turbidity() const { return mTurbidity; }
		void SetTurbidity(float turbidity);

		const float3& GroundAlbedo() const { return mGroundAlbedo; }
		void SetGroundAlbedo(const float3& groundAlbedo);

		// build
		void Build();

		// build info
		const SkyData& CudaData() const { return mCudaData; }
		const SkyState& CudaStateX() const { return mCudaStateX; }
		const SkyState& CudaStateY() const { return mCudaStateY; }
		const SkyState& CudaStateZ() const { return mCudaStateZ; }

	private:
		void FreeStates();
		void SetState(ArHosekSkyModelState& cpu, SkyState& cuda) const;

		// dirty state
		bool mIsDirty = true;

		// sun
		bool mDrawSun = true;
		float3 mSunDir = make_float3(0, 1, 0);
		float3 mSunColor = make_float3(20.f);
		float mSunSize = .27f;

		// tint
		float3 mSkyTint = make_float3(1);
		float3 mSunTint = make_float3(1);

		// ground
		float mTurbidity = 4.f;
		float3 mGroundAlbedo = make_float3(0.18f);

		// Hosek states
		ArHosekSkyModelState* mStateX = nullptr;
		ArHosekSkyModelState* mStateY = nullptr;
		ArHosekSkyModelState* mStateZ = nullptr;

		// CUDA data
		SkyData mCudaData = {};
		SkyState mCudaStateX = {};
		SkyState mCudaStateY = {};
		SkyState mCudaStateZ = {};
	};
}
