#include "CameraWindow.h"

// Project
#include "Gui/GuiHelpers.h"
#include "Resources/CameraNode.h"

// ImGUI
#include "imgui/imgui.h"

namespace Tracer
{
	CameraWindow* const CameraWindow::Get()
	{
		static CameraWindow inst;
		return &inst;
	}



	void CameraWindow::DrawImpl()
	{
		ImGui::Begin("Camera", &mEnabled);
		if(!mCamNode)
		{
			ImGui::Text("No camera node detected");
		}
		else
		{
			bool hasChanged = false;

			// transformation
			float camPos[] = { mCamNode->Position().x, mCamNode->Position().y, mCamNode->Position().z };
			float camTarget[] = { mCamNode->Target().x, mCamNode->Target().y, mCamNode->Target().z };
			float camUp[] = { mCamNode->Up().x, mCamNode->Up().y, mCamNode->Up().z };
			float camFov = mCamNode->Fov() * RadToDeg;

			ImGui::BeginGroup();
			hasChanged = ImGui::InputFloat3("Position", camPos) || hasChanged;
			hasChanged = ImGui::InputFloat3("Target", camTarget) || hasChanged;
			hasChanged = ImGui::InputFloat3("Up", camUp) || hasChanged;
			hasChanged = ImGui::SliderFloat("Fov", &camFov, 1, 179) || hasChanged;
			ImGui::EndGroup();

			if(hasChanged)
			{
				mCamNode->SetPosition(make_float3(camPos[0], camPos[1], camPos[2]));
				mCamNode->SetTarget(make_float3(camTarget[0], camTarget[1], camTarget[2]));
				mCamNode->SetUp(make_float3(camUp[0], camUp[1], camUp[2]));
				mCamNode->SetFov(camFov * DegToRad);
			}
		}

		ImGui::End();
	}
}
