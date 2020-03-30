#pragma once

// Project
#include "Optix/Optix7.h"

// C++
#include <string>



#define OPTIX_CHECK(x) OptixCheck((x), __FILE__, __LINE__)



namespace Tracer
{
	bool InitOptix();
	void OptixCheck(OptixResult res, const char* file, int line);
	std::string ToString(OptixResult optixResult);
}
