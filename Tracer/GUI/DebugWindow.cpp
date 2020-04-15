#include "DebugWindow.h"

// Project
#include "Gui/GuiHelpers.h"

// ImGUI
#include "imgui/imgui.h"

namespace Tracer
{
	DebugWindow* const DebugWindow::Get()
	{
		static DebugWindow inst;
		return &inst;
	}



	void DebugWindow::Set(const std::string& name, const std::string& data)
	{
		mMap[name] = data;
	}



	void DebugWindow::Unset(const std::string& name)
	{
		mMap.erase(name);
	}



	void DebugWindow::DrawImpl()
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
