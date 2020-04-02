#include "Scene.h"

// Project
#include "Resources/Model.h"
#include "Resources/Material.h"

// C++
#include <numeric>

namespace Tracer
{

	size_t Scene::MaterialCount(bool includeModels) const
	{
		return mMaterials.size() + std::accumulate(mModels.begin(), mModels.end(), 0ull, [](size_t a, auto& m) { return a + m->Materials().size(); });
	}



	size_t Scene::MeshCount(bool includeModels) const
	{
		return mMeshes.size() + std::accumulate(mModels.begin(), mModels.end(), 0ull, [](size_t a, auto& m) { return a + m->Meshes().size(); });
	}



	size_t Scene::ModelCount() const
	{
		return mModels.size();
	}



	size_t Scene::TextureCount() const
	{
		size_t cnt = 0;
		for(auto mdl : mModels)
			for(auto mat : mdl->Materials())
				cnt += mat->TextureCount();
		return cnt;
	}



	void Scene::AddMaterial(std::shared_ptr<Material> material)
	{
		if(std::find(mMaterials.begin(), mMaterials.end(), material) == mMaterials.end())
		{
			mMaterials.push_back(material);
			mIsDirty = true;
		}
	}



	void Scene::AddMesh(std::shared_ptr<Mesh> mesh)
	{
		if(std::find(mMeshes.begin(), mMeshes.end(), mesh) == mMeshes.end())
		{
			mMeshes.push_back(mesh);
			mIsDirty = true;
		}
	}



	void Scene::AddModel(std::shared_ptr<Model> model)
	{
		if(std::find(mModels.begin(), mModels.end(), model) == mModels.end())
		{
			mModels.push_back(model);
			mIsDirty = true;
		}
	}
}
