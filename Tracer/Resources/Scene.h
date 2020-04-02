#pragma once

// C++
#include <memory>
#include <vector>

namespace Tracer
{
	class Material;
	class Mesh;
	class Model;
	class Renderer;
	class Scene
	{
		friend class Renderer;
	public:
		size_t MaterialCount(bool includeModels = true) const;
		size_t MeshCount(bool includeModels = true) const;
		size_t ModelCount() const;
		size_t TextureCount() const;

		void AddMaterial(std::shared_ptr<Material> material);
		void AddMesh(std::shared_ptr<Mesh> mesh);
		void AddModel(std::shared_ptr<Model> model);

		inline bool IsDirty() const { return mIsDirty; }
		inline void ResetDirtyFlag() { mIsDirty = false; }

		inline const std::vector<std::shared_ptr<Material>>& Materials() const { return mMaterials; }
		inline const std::vector<std::shared_ptr<Mesh>>& Meshes() const { return mMeshes; }
		inline const std::vector<std::shared_ptr<Model>>& Models() const { return mModels; }

	private:
		// utility
		bool mIsDirty = false;

		// resources
		std::vector<std::shared_ptr<Material>> mMaterials;
		std::vector<std::shared_ptr<Mesh>> mMeshes;
		std::vector<std::shared_ptr<Model>> mModels;
	};
}
