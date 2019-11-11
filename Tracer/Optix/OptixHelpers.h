#pragma once

#include "Optix7.h"

// C++
#include <string>

namespace Tracer
{
	/*!
	 *
	 */
	class OptixHelpers
	{
	public:
		/*!
		*/
		static bool Init();


		/*!
		*/
		static std::string ToString(OptixResult optixResult);
	};
}
