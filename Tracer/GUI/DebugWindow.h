#pragma once

// Project
#include "GuiWindow.h"

// C++
#include <string>
#include <map>

namespace Tracer
{
	class DebugWindow : public GuiWindow
	{
	public:
		static DebugWindow* const Get();

		void Set(const std::string& name, const std::string& data);
		void Unset(const std::string& name);

	private:
		void DrawImpl() final;

		std::map<std::string, std::string> mMap;
	};
}
