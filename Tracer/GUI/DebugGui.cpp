#include "DebugGui.h"

// Project
#include "Gui/GuiHelpers.h"

// ImGUI
#include "imgui/imgui.h"

namespace Tracer
{
	DebugGui* const DebugGui::Get()
	{
		static DebugGui inst;
		return &inst;
	}



	void DebugGui::Set(const std::string& name, const std::string& data)
	{
		mMap[name] = data;
	}



	void DebugGui::Unset(const std::string& name)
	{
		mMap.erase(name);
	}



	void DebugGui::DrawImpl()
	{
		ImGui::Begin("Debug", &mEnabled);

		ImGui::Columns(2);

		// table header
		ImGui::Separator();
		ImGui::Text("Data");
		ImGui::NextColumn();
		ImGui::Text("Value");
		ImGui::NextColumn();
		ImGui::Separator();

		// data
		for(auto& kv : mMap)
		{
			ImGui::Text(kv.first.c_str());
			ImGui::NextColumn();
			ImGui::Text(kv.second.c_str());
			ImGui::NextColumn();
		}

		ImGui::End();
	}
}
