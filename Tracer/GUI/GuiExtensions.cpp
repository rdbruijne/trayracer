#include "GuiExtensions.h"

// C++
#include <unordered_map>

namespace ImGui
{
	bool SliderUInt(const char* label, unsigned int* v, unsigned int v_min, unsigned int v_max, const char* format, ImGuiSliderFlags flags)
	{
		return SliderScalar(label, ImGuiDataType_U32, v, &v_min, &v_max, format, flags);
	}



	bool SliderUInt2(const char* label, unsigned int v[2], unsigned int v_min, unsigned int v_max, const char* format, ImGuiSliderFlags flags)
	{
		return SliderScalarN(label, ImGuiDataType_U32, v, 2, &v_min, &v_max, format, flags);
	}



	bool SliderUInt3(const char* label, unsigned int v[3], unsigned int v_min, unsigned int v_max, const char* format, ImGuiSliderFlags flags)
	{
		return SliderScalarN(label, ImGuiDataType_U32, v, 3, &v_min, &v_max, format, flags);
	}



	bool SliderUInt4(const char* label, unsigned int v[4], unsigned int v_min, unsigned int v_max, const char* format, ImGuiSliderFlags flags)
	{
		return SliderScalarN(label, ImGuiDataType_U32, v, 4, &v_min, &v_max, format, flags);
	}



	bool InputUInt(const char* label, unsigned int* v, unsigned int step, unsigned int step_fast, ImGuiInputTextFlags flags)
	{
		// Hexadecimal input provided as a convenience but the flag name is awkward. Typically you'd use InputText() to parse your own data, if you want to handle prefixes.
		const char* format = (flags & ImGuiInputTextFlags_CharsHexadecimal) ? "%08X" : "%d";
		return InputScalar(label, ImGuiDataType_U32, (void*)v, (void*)(step > 0 ? &step : NULL), (void*)(step_fast > 0 ? &step_fast : NULL), format, flags);
	}



	bool InputUInt2(const char* label, unsigned int v[2], ImGuiInputTextFlags flags)
	{
		return InputScalarN(label, ImGuiDataType_U32, v, 2, NULL, NULL, "%d", flags);
	}



	bool InputUInt3(const char* label, unsigned int v[3], ImGuiInputTextFlags flags)
	{
		return InputScalarN(label, ImGuiDataType_U32, v, 3, NULL, NULL, "%d", flags);
	}



	bool InputUInt4(const char* label, unsigned int v[4], ImGuiInputTextFlags flags)
	{
		return InputScalarN(label, ImGuiDataType_U32, v, 4, NULL, NULL, "%d", flags);
	}



	bool ComboBox(const std::string& name, int& value, const std::vector<std::string>& keys)
	{
		bool changed = false;
		const std::string propName = (value >= 0 && value < static_cast<int>(keys.size())) ? keys[static_cast<size_t>(value)] : "";
		if(BeginCombo(name.c_str(), propName.c_str()))
		{
			for(int i = 0; i < static_cast<int>(keys.size()); i++)
			{
				const std::string itemName = keys[static_cast<size_t>(i)];
				if(Selectable(itemName.c_str(), i == value))
				{
					value = i;
					changed = true;
				}

				if(i == value)
				{
					SetItemDefaultFocus();
				}
			}
			EndCombo();
		}
		return changed;
	}
}
