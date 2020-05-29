#include "MaterialGui.h"

// Project
#include "FileIO/Importer.h"
#include "Gui/GuiHelpers.h"
#include "Resources/Material.h"
#include "Utility/Utility.h"

// ImGUI
#include "imgui/imgui.h"

// C++
#include <functional>



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
		if(mSelectedMaterial.expired() || mSelectedMaterial.use_count() == 0)
		{
			ImGui::Text("No material detected");
		}
		else
		{
			auto mat = mSelectedMaterial.lock();
			ImGui::Text(mat->Name().c_str());

			float3 diff = mat->Diffuse();
			if(ImGui::ColorEdit3("Diffuse", reinterpret_cast<float*>(&diff)))
				mat->SetDiffuse(diff);

			float3 em = mat->Emissive();
			if(ImGui::ColorEdit3("Emissive", reinterpret_cast<float*>(&em), ImGuiColorEditFlags_HDR))
				mat->SetEmissive(em);

			DrawTexture("Diffuse map", [=]() { return mat->DiffuseMap(); }, [=](std::shared_ptr<Texture> a) { mat->SetDiffuseMap(a); });

			DrawTexture("Normal map", [=]() { return mat->NormalMap(); }, [=](std::shared_ptr<Texture> a) { mat->SetNormalMap(a); });
		}
		ImGui::End();
	}



	void MaterialGui::DrawTexture(const std::string& name, std::function<std::shared_ptr<Texture>()> getTex, std::function<void(std::shared_ptr<Texture>)> setTex)
	{
		ImGui::Spacing();
		ImGui::Spacing();

		auto tex = getTex();
		if(!tex)
		{
			ImGui::Text("%s: N/A", name.c_str());
			if(ImGui::Button("Load texture"))
			{
				std::string texFile;
				if(OpenFileDialog("Png\0*.png\0", "Select a texture file", true, texFile))
					setTex(Importer::ImportTexture(mScene, texFile));
			}
		}
		else
		{
			tex->MakeGlTex();

			// tex ID
			size_t texId = static_cast<size_t>(tex->GLTex()->ID());

			// resolution
			const ImVec2 res = ImVec2(100, 100);

			// display
			const float dpiScale = ImGui::GetIO().FontGlobalScale;
			ImGui::Text("%s: %s", name.c_str(), tex->Name().c_str());
			//ImGui::Image(reinterpret_cast<ImTextureID>(texId), ImVec2(res.x * dpiScale, res.y * dpiScale));
			if(ImGui::ImageButton(reinterpret_cast<ImTextureID>(texId), ImVec2(res.x * dpiScale, res.y * dpiScale)))
			{
				std::string texFile;
				if(OpenFileDialog("Png\0*.png\0", "Select a texture file", true, texFile))
					setTex(Importer::ImportTexture(mScene, texFile));
			}
		}
	}
}
