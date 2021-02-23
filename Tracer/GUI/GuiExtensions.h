#pragma once

// Magic Enum
#pragma warning(push)
#pragma warning(disable: 4346 5027)
#include "magic_enum/magic_enum.hpp"
#pragma warning(pop)

// ImGUI
#include "imgui/imgui.h"

// C++
#include <string>

namespace Tracer
{
	// enum combo box
	template<typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value, Enum>::type>
	static bool ComboBox(const std::string& name, Enum& value)
	{
		bool changed = false;
		const std::string propName = std::string(magic_enum::enum_name(value));
		if(ImGui::BeginCombo(name.c_str(), propName.c_str()))
		{
			for(size_t i = 0; i < magic_enum::enum_count<Enum>(); i++)
			{
				const Enum e = static_cast<Enum>(i);
				const std::string itemName = std::string(magic_enum::enum_name(e));
				if(ImGui::Selectable(itemName.c_str(), e == value))
				{
					value = e;
					changed = true;
				}

				if(e == value)
				{
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndCombo();
		}
		return changed;
	}
}
