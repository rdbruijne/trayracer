#pragma once

// C++
#include <string>

namespace Tracer
{
	class Shader;
	class ShaderDescriptor
	{
	public:
		static bool Load(const std::string& descriptorFile, Shader* shader);
		static bool Save(const std::string& descriptorFile, Shader* shader);
	};
}
