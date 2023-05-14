#pragma once

// Magic Enum
#pragma warning(push)
#pragma warning(disable: 4346) // 'name' : dependent name is not a type
#pragma warning(disable: 4626) // 'derived class' : assignment operator was implicitly defined as deleted because a base class assignment operator is inaccessible or deleted
#pragma warning(disable: 5027) // 'type': move assignment operator was implicitly defined as deleted
#include "magic_enum/magic_enum.hpp"
#pragma warning(pop)

// ImGUI
#include "imgui/imgui.h"

// C++
#include <string>

namespace Tracer
{
	// enum combo box
	template<typename Enum, std::enable_if_t<std::is_enum_v<Enum>, bool> = true>
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



	static bool ComboBox(const std::string& name, int& value, const std::vector<std::string>& keys)
	{
		bool changed = false;
		const std::string propName = (value >= 0 && value < static_cast<int>(keys.size())) ? keys[static_cast<size_t>(value)] : "";
		if(ImGui::BeginCombo(name.c_str(), propName.c_str()))
		{
			for(int i = 0; i < static_cast<int>(keys.size()); i++)
			{
				const std::string itemName = keys[static_cast<size_t>(i)];
				if(ImGui::Selectable(itemName.c_str(), i == value))
				{
					value = i;
					changed = true;
				}

				if(i == value)
				{
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndCombo();
		}
		return changed;
	}
}
