#include "RendererWindow.h"

// Project
#include "Optix/Renderer.h"
#include "Gui/GuiHelpers.h"

// ImGUI
#include "imgui/imgui.h"

// magic enum
#include "magic_enum/magic_enum.hpp"

namespace Tracer
{
	void RendererWindow::Draw()
	{
		// get the camera
		Renderer* renderer = GuiHelpers::renderer;

		ImGui::Begin("Renderer", &mEnabled);
		if(!renderer)
		{
			ImGui::Text("No renderer node detected");
			ImGui::End();
			return;
		}

		//bool hasChanged = false;

		// render mode
		Renderer::RenderModes activeRenderMode = renderer->GetRenderMode();
		const std::string rmName = ToString(activeRenderMode);
		if(ImGui::BeginCombo("Render Mode", rmName.c_str()))
		{
			for(size_t i = 0; i <magic_enum::enum_count<Renderer::RenderModes>(); i++)
			{
				const Renderer::RenderModes mode = static_cast<Renderer::RenderModes>(i);
				const std::string itemName = ToString(mode);
				if(ImGui::Selectable(itemName.c_str(), mode == activeRenderMode))
					renderer->SetRenderMode(mode);
				if(mode == activeRenderMode)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}
		
		ImGui::End();
	}
}
