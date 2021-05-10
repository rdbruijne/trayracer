#pragma once

#include <stdint.h>

namespace Tracer
{
	struct RenderStatistics
	{
		uint64_t pathCount = 0;

		uint64_t primaryPathCount = 0;
		uint64_t secondaryPathCount = 0;
		uint64_t deepPathCount = 0;
		uint64_t shadowRayCount = 0;

		float renderTimeMs = 0;
		float primaryPathTimeMs = 0;
		float secondaryPathTimeMs = 0;
		float deepPathTimeMs = 0;
		float shadowTimeMs = 0;
		float shadeTimeMs = 0;
		float denoiseTimeMs = 0;

		float buildTimeMs = 0;
		float geoBuildTimeMs = 0;
		float matBuildTimeMs = 0;
		float skyBuildTimeMs = 0;
	};
}