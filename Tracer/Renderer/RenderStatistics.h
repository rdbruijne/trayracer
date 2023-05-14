#pragma once

#include <stdint.h>
#include <vector>

namespace Tracer
{
	struct RenderStatistics
	{
		// device statistics
		struct DeviceStatistics
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
		};
		DeviceStatistics device;

		// render times
		float renderTimeMs = 0;
		float denoiseTimeMs = 0;

		// total build time
		float buildTimeMs = 0;

		// detailed build times
		float geoBuildTimeMs = 0;
		float matBuildTimeMs = 0;
		float skyBuildTimeMs = 0;

		// detailed upload times
		float geoUploadTimeMs = 0;
		float matUploadTimeMs = 0;
		float skyUploadTimeMs = 0;

		
	};
}
