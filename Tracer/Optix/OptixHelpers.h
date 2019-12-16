#pragma once

// OptiX
#include "Optix7.h"

// C++
#include <cassert>
#include <string>
#include <sstream>



/*!
 * @brief Check the value of an OptixResult. Will throw an exception when an error occurs.
 */
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
	/*!
	* @brief Initialize OptiX.
	* @return True on success, false otherwise.
	*/
	bool InitOptix();


	/*!
	* @brief Convert OptixResult to corresponding string.
	* @param[in] optixResult OptixResult code to convert.
	* @return String containing error code.
	*/
	std::string ToString(OptixResult optixResult);
}
