#include "Scene.h"

// Project
#include "Resources/Model.h"
#include "Resources/Instance.h"
#include "Resources/Material.h"

// C++
#include <set>

namespace Tracer
{
	bool Scene::IsDirty() const
	{
		if(mIsDirty)
			return true;

		for(auto& i : mInstances)
			if(i->IsDirty())
				return true;

		return false;
	}



	size_t Scene::InstanceCount() const
	{
		return mInstances.size();
	}



	size_t Scene::MaterialCount() const
	{
		size_t matCount = 0;
		for(auto m : mModels)
			matCount += m->Materials().size();
		return matCount;
	}



	size_t Scene::ModelCount() const
	{
		return mModels.size();
	}



	size_t Scene::InstancedModelCount() const
	{
		std::set<std::shared_ptr<Model>> models;
		for(auto& i : mInstances)
			models.insert(i->GetModel());
		return models.size();
	}



	size_t Scene::TriangleCount() const
	{
		size_t triCount = 0;
		for(auto& i : mInstances)
			triCount += i->GetModel()->PolyCount();
		return triCount;
	}



	size_t Scene::UniqueTriangleCount() const
	{
		std::set<std::shared_ptr<Model>> models;
		for(auto& i : mInstances)
			models.insert(i->GetModel());

		size_t triCount = 0;
		for(const auto& m : models)
			triCount += m->PolyCount();
		return triCount;
	}



	void Scene::AddModel(std::shared_ptr<Model> model)
	{
		if(model && std::find(mModels.begin(), mModels.end(), model) == mModels.end())
		{
			mModels.push_back(model);
			MarkDirty();
		}
	}



	void Scene::AddInstance(std::shared_ptr<Instance> instance)
	{
		assert(instance->GetModel() != nullptr);
		assert(std::find(mModels.begin(), mModels.end(), instance->GetModel()) != mModels.end());

		if(instance && std::find(mInstances.begin(), mInstances.end(), instance) == mInstances.end())
		{
			mInstances.push_back(instance);
			MarkDirty();
		}
	}



	std::shared_ptr<Tracer::Material> Scene::GetMaterial(uint32_t instanceIx, uint32_t primIx)
	{
		if(instanceIx >= mInstances.size())
			return nullptr;
		return mInstances[instanceIx]->GetModel()->GetMaterial(primIx);
	}
}
