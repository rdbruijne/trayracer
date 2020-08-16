#include "MaterialGui.h"

// Project
#include "FileIO/Importer.h"
#include "GUI/GuiHelpers.h"
#include "Resources/Material.h"
#include "Utility/Utility.h"

// ImGUI
#include "imgui/imgui.h"

// C++
#include <functional>



namespace Tracer
{
	namespace
	{
		std::string ImportTextureDialog()
		{
			std::string filter = "";
			if(filter.empty())
			{
				const std::vector<Importer::Format>& texFormats = Importer::SupportedTextureFormats();
				std::string extensions = "";
				for(auto f : texFormats)
				{
					std::vector<std::string> extParts = Split(f.ext, ',');
					for(auto e : extParts)
						extensions += std::string(extensions.empty() ? "" : ";") + "*." + e;
				}

				const std::string& zeroString = std::string(1, '\0');
				filter = "Image files" + zeroString + extensions + zeroString;
			}

			std::string texFile = "";
			OpenFileDialog(filter.c_str(), "Select a texture file", true, texFile);
			return texFile;
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
		// #TODO: decent layount

		ImGui::Spacing();
		ImGui::Spacing();

		auto tex = getTex();
		if(!tex)
		{
			ImGui::Text("%s: N/A", name.c_str());
			const std::string buttonName = format("Load texture##%s", name.c_str());
			if(ImGui::Button(buttonName.c_str()))
			{
				std::string texFile = ImportTextureDialog();
				if(!texFile.empty())
					setTex(Importer::ImportTexture(GuiHelpers::scene, texFile));
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
			ImGui::Text("%s", name.c_str(), tex->Name().c_str());
			ImGui::Columns(3);

			ImGui::Text(tex->Name().c_str());
			ImGui::NextColumn();

			if(ImGui::ImageButton(reinterpret_cast<ImTextureID>(texId), ImVec2(res.x * dpiScale, res.y * dpiScale)))
			{
				std::string texFile = ImportTextureDialog();
				if(!texFile.empty())
					setTex(Importer::ImportTexture(GuiHelpers::scene, texFile));
			}
			ImGui::NextColumn();

			// remove texture button
			const std::string buttonName = format("Remove##%s", name.c_str());
			if(ImGui::Button(buttonName.c_str()))
			   setTex(nullptr);

			ImGui::Columns(1);
		}
	}
}
