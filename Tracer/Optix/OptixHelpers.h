#pragma once

// OptiX
#include "Optix7.h"

// C++
#include <string>

namespace Tracer
{
	/*!
	 * @brief Helper class for OptiX functionality.
	 */
	class OptixHelpers
	{
	public:
		/*!
		 * @brief Initialize OptiX
		 * @return True on success, false otherwise.
		 */
		static bool Init();


		/*!
		 * @brief Convert OptixResult to corresponding string.
		 * @param[in] optixResult OptixResult code to convert.
		 * @return String containing error code.
		 */
		static std::string ToString(OptixResult optixResult);
	};
}
