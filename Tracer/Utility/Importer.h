#pragma once

// C++
#include <memory>
#include <string>

namespace Tracer
{
	class Model;
	std::shared_ptr<Model> ImportModel(const std::string& filePath);
}
