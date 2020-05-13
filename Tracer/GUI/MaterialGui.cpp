#include "MaterialGui.h"

// Project
#include "Gui/GuiHelpers.h"
#include "Resources/Material.h"

// ImGUI
#include "imgui/imgui.h"

namespace Tracer
{
	MaterialGui* const MaterialGui::Get()
	{
		static MaterialGui inst;
		return &inst;
	}



	void MaterialGui::DrawImpl()
	{
		ImGui::Begin("Material", &mEnabled);
		if(selectedMaterial.expired() || selectedMaterial.use_count() == 0)
		{
			ImGui::Text("No material detected");
		}
		else
		{
			auto mat = selectedMaterial.lock();
			ImGui::Text(mat->Name().c_str());

			float3 diff = mat->Diffuse();
			if(ImGui::ColorEdit3("Diffuse", reinterpret_cast<float*>(&diff)))
				mat->SetDiffuse(diff);

			float3 em = mat->Emissive();
			if(ImGui::ColorEdit3("Emissive", reinterpret_cast<float*>(&em), ImGuiColorEditFlags_HDR))
				mat->SetEmissive(em);
		}

		ImGui::End();
	}
}
