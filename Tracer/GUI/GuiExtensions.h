#pragma once

// Magic Enum
#include "magic_enum/magic_enum.hpp"

// ImGUI
#include "imgui/imgui.h"

// C++
#include <string>

namespace ImGui
{
	// Widgets: Regular Sliders
	bool SliderUInt(const char* label, unsigned int* v, unsigned int v_min, unsigned int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0);
	bool SliderUInt2(const char* label, unsigned int v[2], unsigned int v_min, unsigned int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0);
	bool SliderUInt3(const char* label, unsigned int v[3], unsigned int v_min, unsigned int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0);
	bool SliderUInt4(const char* label, unsigned int v[4], unsigned int v_min, unsigned int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0);

	// Widgets: Input with Keyboard
	bool InputUInt(const char* label, unsigned int* v, unsigned int step = 1, unsigned int step_fast = 100, ImGuiInputTextFlags flags = 0);
	bool InputUInt2(const char* label, unsigned int v[2], ImGuiInputTextFlags flags = 0);
	bool InputUInt3(const char* label, unsigned int v[3], ImGuiInputTextFlags flags = 0);
	bool InputUInt4(const char* label, unsigned int v[4], ImGuiInputTextFlags flags = 0);

	// combo box
	bool ComboBox(const std::string& name, int& value, const std::vector<std::string>& keys);

	// enum combo box
	template<typename Enum, std::enable_if_t<std::is_enum_v<Enum>, bool> = false>
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
