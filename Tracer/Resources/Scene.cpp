#include "Scene.h"

// Project
#include "Optix/OptixHelpers.h"
#include "Optix/Renderer.h"

namespace Tracer
{
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
}
