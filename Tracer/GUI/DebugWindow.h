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
		static void Set(const std::string& name, const std::string& data);
		static void Unset(const std::string& name);

	private:
		void DrawImpl() final;

		static std::map<std::string, std::string> msMap;
	};
}
