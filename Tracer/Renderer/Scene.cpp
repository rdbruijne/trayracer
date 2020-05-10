#include "Scene.h"

// Project
#include "Resources/Model.h"
#include "Resources/Instance.h"
#include "Resources/Material.h"

namespace Tracer
{

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



	bool Scene::IsDirty() const
	{
		if(mIsDirty)
			return true;

		for(auto& i : mInstances)
			if(i->IsDirty())
				return true;

		return false;
	}

}
