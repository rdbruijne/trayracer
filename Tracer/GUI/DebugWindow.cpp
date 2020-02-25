#include "DebugWindow.h"

// Project
#include "Gui/GuiHelpers.h"

// ImGUI
#include "imgui/imgui.h"

namespace Tracer
{
	std::map<std::string, std::string> DebugWindow::msMap;

	void DebugWindow::Draw()
	{
		ImGui::Begin("Debug", &mEnabled);
	
		for(auto& kv : msMap)
			ImGui::LabelText(kv.first.c_str(), kv.second.c_str());

		ImGui::End();
	}



	void DebugWindow::Set(const std::string& name, const std::string& data)
	{
		msMap[name] = data;
	}



	void DebugWindow::Unset(const std::string& name)
	{
		msMap.erase(name);
	}
}
