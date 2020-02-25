#include "CameraWindow.h"

// Project
#include "App/CameraNode.h"
#include "Gui/GuiHelpers.h"

// ImGUI
#include "imgui/imgui.h"

namespace Tracer
{
	void CameraWindow::Draw()
	{
		// get the camera
		CameraNode* camNode = GuiHelpers::CamNode;

		ImGui::Begin("Camera", &mEnabled);
		if(!camNode)
		{
			ImGui::Text("No camera node detected");
			ImGui::End();
			return;
		}

		bool hasChanged = false;

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

		ImGui::End();
	}
}
