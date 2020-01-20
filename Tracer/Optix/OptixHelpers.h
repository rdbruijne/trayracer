#pragma once

// Project
#include "Optix/Optix7.h"

// C++
#include <cassert>
#include <string>
#include <sstream>



#define OPTIX_CHECK(x)																											\
	{																															\
		OptixResult res = x;																									\
		assert(res == OPTIX_SUCCESS);																							\
		if(res != OPTIX_SUCCESS)																								\
		{																														\
			std::stringstream ss;																								\
			ss << "OptiX call \"" << #x << "\"failed with error code " << res;													\
			throw std::runtime_error(ss.str());																					\
		}																														\
	}



namespace Tracer
{
	bool InitOptix();
	std::string ToString(OptixResult optixResult);
}
