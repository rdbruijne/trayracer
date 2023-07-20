#pragma once

#include <cstdint>
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



	// helpers for statistics accumulation
	inline RenderStatistics::DeviceStatistics operator + (const RenderStatistics::DeviceStatistics& a, const RenderStatistics::DeviceStatistics& b)
	{
		RenderStatistics::DeviceStatistics c;

		c.pathCount = a.pathCount + b.pathCount;

		c.primaryPathCount = a.primaryPathCount + b.primaryPathCount;
		c.secondaryPathCount = a.secondaryPathCount + b.secondaryPathCount;
		c.deepPathCount = a.deepPathCount + b.deepPathCount;
		c.shadowRayCount = a.shadowRayCount + b.shadowRayCount;

		c.renderTimeMs = a.renderTimeMs + b.renderTimeMs;
		c.primaryPathTimeMs = a.primaryPathTimeMs + b.primaryPathTimeMs;
		c.secondaryPathTimeMs = a.secondaryPathTimeMs + b.secondaryPathTimeMs;
		c.deepPathTimeMs = a.deepPathTimeMs + b.deepPathTimeMs;
		c.shadowTimeMs = a.shadowTimeMs + b.shadowTimeMs;
		c.shadeTimeMs = a.shadeTimeMs + b.shadeTimeMs;

		return c;
	}



	inline void operator += (RenderStatistics::DeviceStatistics& a, const RenderStatistics::DeviceStatistics& b)
	{
		a = a + b;
	}



	inline RenderStatistics operator + (const RenderStatistics& a, const RenderStatistics& b)
	{
		RenderStatistics c;

		// device statistics
		c.device = a.device + b.device;

		// render times
		c.renderTimeMs = a.renderTimeMs + b.renderTimeMs;
		c.denoiseTimeMs = a.denoiseTimeMs + b.denoiseTimeMs;

		// total build time
		c.buildTimeMs = a.buildTimeMs + b.buildTimeMs;

		// detailed build times
		c.geoBuildTimeMs = a.geoBuildTimeMs + b.geoBuildTimeMs;
		c.matBuildTimeMs = a.matBuildTimeMs + b.matBuildTimeMs;
		c.skyBuildTimeMs = a.skyBuildTimeMs + b.skyBuildTimeMs;

		// detailed upload times
		c.geoUploadTimeMs = a.geoUploadTimeMs + b.geoUploadTimeMs;
		c.matUploadTimeMs = a.matUploadTimeMs + b.matUploadTimeMs;
		c.skyUploadTimeMs = a.skyUploadTimeMs + b.skyUploadTimeMs;

		return c;
	}



	inline RenderStatistics& operator += (RenderStatistics& a, const RenderStatistics& b)
	{
		a = a + b;
		return a;
	}
}
