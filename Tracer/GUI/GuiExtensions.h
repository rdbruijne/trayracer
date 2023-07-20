#pragma once

// Magic Enum
#include "magic_enum/magic_enum.hpp"

// ImGUI
#include "imgui/imgui.h"

// C++
#include <string>

namespace ImGui
{
	// slider widgets (adapted from ImGui::SliderInt<N>)
	bool SliderUInt(const char* label, unsigned int* v, unsigned int v_min, unsigned int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0);
	bool SliderUInt2(const char* label, unsigned int v[2], unsigned int v_min, unsigned int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0);
	bool SliderUInt3(const char* label, unsigned int v[3], unsigned int v_min, unsigned int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0);
	bool SliderUInt4(const char* label, unsigned int v[4], unsigned int v_min, unsigned int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0);

	// input widgets (adapted from ImGui::InputInt<N>)
	bool InputUInt(const char* label, unsigned int* v, unsigned int step = 1, unsigned int step_fast = 100, ImGuiInputTextFlags flags = 0);
	bool InputUInt2(const char* label, unsigned int v[2], ImGuiInputTextFlags flags = 0);
	bool InputUInt3(const char* label, unsigned int v[3], ImGuiInputTextFlags flags = 0);
	bool InputUInt4(const char* label, unsigned int v[4], ImGuiInputTextFlags flags = 0);

	// combo box
	bool ComboBox(const std::string& name, int& value, const std::vector<std::string>& keys);

	template<typename Enum, std::enable_if_t<std::is_enum_v<Enum>, bool> = false>
	static bool ComboBox(const std::string& name, Enum& value);
}

#include "GuiExtensions.inl"
