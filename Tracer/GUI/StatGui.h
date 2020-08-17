#pragma once

// Project
#include "BaseGui.h"

// C++
#include <array>
#include <stdint.h>

namespace Tracer
{
	class Renderer;
	class Scene;
	class StatGui : public BaseGui
	{
	public:
		static StatGui* const Get();

		float mFrameTimeMs = 0;
		float mBuildTimeMs = 0;

	private:
		void DrawImpl() final;

		static constexpr size_t msGraphSize = 128;

		size_t mGraphIx = 0;

		std::array<float, msGraphSize> mFramerates;
		std::array<float, msGraphSize> mFrameTimes;
		std::array<float, msGraphSize> mBuildTimes;
		std::array<float, msGraphSize> mPrimaryPathTimes;
		std::array<float, msGraphSize> mSecondaryPathTimes;
		std::array<float, msGraphSize> mDeepPathTimes;
		std::array<float, msGraphSize> mShadowTimes;
		std::array<float, msGraphSize> mShadeTimes;
		std::array<float, msGraphSize> mDenoiseTimes;

		std::array<float, msGraphSize> mPathCounts;
		std::array<float, msGraphSize> mPrimaryPathCounts;
		std::array<float, msGraphSize> mSecondaryPathCounts;
		std::array<float, msGraphSize> mDeepPathCounts;
		std::array<float, msGraphSize> mShadowRayCounts;
	};
}
