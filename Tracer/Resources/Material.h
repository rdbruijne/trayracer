#pragma once

// Project
#include "Utility/LinearMath.h"

// C++
#include <string>

namespace Tracer
{
	class Material
	{
	public:
		explicit Material(const std::string& name);

		std::string mName = "";

		// material properties
		float3 mDiffuse = make_float3(.5f, .5f, .5f);
	};
}
