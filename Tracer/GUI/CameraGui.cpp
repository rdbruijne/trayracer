#include "CameraGui.h"

// Project
#include "GUI/GuiHelpers.h"
#include "Resources/CameraNode.h"

// ImGUI
#include "imgui/imgui.h"

namespace Tracer
{
	CameraGui* const CameraGui::Get()
	{
		static CameraGui inst;
		return &inst;
	}



	void CameraGui::DrawImpl()
	{
		ImGui::Begin("Camera", &mEnabled);
		if(!GuiHelpers::camNode)
		{
			ImGui::Text("No camera node detected");
		}
		else
		{
			bool hasChanged = false;

			// transformation
			float camPos[] = { GuiHelpers::camNode->Position().x, GuiHelpers::camNode->Position().y, GuiHelpers::camNode->Position().z };
			float camTarget[] = { GuiHelpers::camNode->Target().x, GuiHelpers::camNode->Target().y, GuiHelpers::camNode->Target().z };
			float camUp[] = { GuiHelpers::camNode->Up().x, GuiHelpers::camNode->Up().y, GuiHelpers::camNode->Up().z };
			float camFov = GuiHelpers::camNode->Fov() * RadToDeg;

			ImGui::BeginGroup();
			hasChanged = ImGui::InputFloat3("Position", camPos) || hasChanged;
			hasChanged = ImGui::InputFloat3("Target", camTarget) || hasChanged;
			hasChanged = ImGui::InputFloat3("Up", camUp) || hasChanged;
			hasChanged = ImGui::SliderFloat("Fov", &camFov, 1, 179) || hasChanged;
			ImGui::EndGroup();

			if(hasChanged)
			{
				GuiHelpers::camNode->SetPosition(make_float3(camPos[0], camPos[1], camPos[2]));
				GuiHelpers::camNode->SetTarget(make_float3(camTarget[0], camTarget[1], camTarget[2]));
				GuiHelpers::camNode->SetUp(make_float3(camUp[0], camUp[1], camUp[2]));
				GuiHelpers::camNode->SetFov(camFov * DegToRad);
			}
		}

		ImGui::End();
	}
}
