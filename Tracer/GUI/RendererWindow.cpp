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
		const std::string rmName = std::string(magic_enum::enum_name(activeRenderMode).data());
		if(ImGui::BeginCombo("Render Mode", rmName.c_str()))
		{
			for(size_t i = 0; i <magic_enum::enum_count<Renderer::RenderModes>(); i++)
			{
				const Renderer::RenderModes mode = static_cast<Renderer::RenderModes>(i);
				const std::string itemName = std::string(magic_enum::enum_name(mode).data());
				if(ImGui::Selectable(itemName.c_str(), mode == activeRenderMode))
					renderer->SetRenderMode(mode);
				if(mode == activeRenderMode)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}

#if false
		// transformation
		float camPos[] = { camNode->Position.x, camNode->Position.y, camNode->Position.z };
		float camTarget[] = { camNode->Target.x, camNode->Target.y, camNode->Target.z };
		float camUp[] = { camNode->Up.x, camNode->Up.y, camNode->Up.z };
		float camFov = camNode->Fov * RadToDeg;

		ImGui::BeginGroup();
		hasChanged = ImGui::InputFloat3("Position", camPos) || hasChanged;
		hasChanged = ImGui::InputFloat3("Target", camTarget) || hasChanged;
		hasChanged = ImGui::InputFloat3("Up", camUp) || hasChanged;
		hasChanged = ImGui::InputFloat("Fov", &camFov) || hasChanged;
		ImGui::EndGroup();

		if(hasChanged)
		{
			camNode->Position = make_float3(camPos[0], camPos[1], camPos[2]);
			camNode->Target = make_float3(camTarget[0], camTarget[1], camTarget[2]);
			camNode->Up = make_float3(camUp[0], camUp[1], camUp[2]);
			camNode->Fov = camFov * DegToRad;
		}
#endif

		ImGui::End();
	}
}
