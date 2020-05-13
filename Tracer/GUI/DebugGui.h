#pragma once

// Project
#include "BaseGui.h"

// C++
#include <string>
#include <map>

namespace Tracer
{
	class DebugGui : public BaseGui
	{
	public:
		static DebugGui* const Get();

		void Set(const std::string& name, const std::string& data);
		void Unset(const std::string& name);

	private:
		void DrawImpl() final;

		std::map<std::string, std::string> mMap;
	};
}
