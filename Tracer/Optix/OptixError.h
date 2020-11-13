#pragma once

// Optix
#include "optix7/optix.h"

// C++
#include <string>

#define OPTIX_CHECK(x) OptixCheck((x), __FILE__, __LINE__)

namespace Tracer
{
	void OptixCheck(OptixResult res, const char* file, int line);

	std::string ToString(OptixResult optixResult);
}
