#include "MaterialGui.h"

// Project
#include "Gui/GuiHelpers.h"
#include "Resources/Material.h"

// ImGUI
#include "imgui/imgui.h"

namespace Tracer
{
	namespace
	{
		void DrawTexture(const std::string& name, std::shared_ptr<Texture> tex)
		{
			if(!tex)
			{
				ImGui::Text("%s: N/A", name.c_str());
				return;
			}

			tex->MakeGlTex();

			// tex ID
			size_t texId = static_cast<size_t>(tex->GLTex()->ID());

			// resolution
			const ImVec2 res = ImVec2(100, 100);

			// display
			const float dpiScale = ImGui::GetIO().FontGlobalScale;
			ImGui::Text("%s: %s", name.c_str(), tex->Name().c_str());
			ImGui::Image(reinterpret_cast<ImTextureID>(texId), ImVec2(res.x * dpiScale, res.y * dpiScale));
		}
	}



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

			DrawTexture("Diffuse map", mat->DiffuseMap());
		}

		ImGui::End();
	}
}
