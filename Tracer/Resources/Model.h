#pragma once

// C++
#include <memory>
#include <string>
#include <vector>

namespace Tracer
{
	class Material;
	class Mesh;
	class Model
	{
	public:
		Model() = default;
		explicit Model(const std::string& name);

		void AddMaterial(std::shared_ptr<Material> mat) { mMaterials.push_back(mat); }
		void AddMesh(std::shared_ptr<Mesh> mesh) { mMeshes.push_back(mesh); }

		const std::vector<std::shared_ptr<Material>>& Materials() const { return mMaterials; }
		const std::vector<std::shared_ptr<Mesh>>& Meshes() const { return mMeshes; }

	private:
		std::string mName = "";
		std::vector<std::shared_ptr<Material>> mMaterials;
		std::vector<std::shared_ptr<Mesh>> mMeshes;
	};
}
