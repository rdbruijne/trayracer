#include "Scene.h"

// Project
#include "Resources/Model.h"
#include "Resources/Instance.h"
#include "Resources/Material.h"

// C++
#include <map>
#include <set>
#include <string>

namespace Tracer
{

	void Scene::Clear()
	{
		mModels.clear();
		mInstances.clear();
		MarkDirty();
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



	size_t Scene::LightCount() const
	{
		size_t lightCount = 0;
		for(auto& i : mInstances)
			lightCount += i->GetModel()->LightCount();
		return lightCount;
	}



	size_t Scene::UniqueLightCount() const
	{
		std::set<std::shared_ptr<Model>> models;
		for(auto& i : mInstances)
			models.insert(i->GetModel());

		size_t lightCount = 0;
		for(const auto& m : models)
			lightCount += m->LightCount();
		return lightCount;
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



	std::vector<LightTriangle> Scene::Lights() const
	{
		// #TODO: cache?
		std::vector<LightTriangle> lights;
		size_t lightSize = 0;
		for(size_t i = 0; i < mInstances.size(); i++)
		{
			auto inst = mInstances[i];
			const std::vector<LightTriangle>& modelLights = inst->GetModel()->Lights();
			const float3x4& trans = inst->Transform();
			if(modelLights.size() > 0)
			{
				lights.insert(lights.end(), modelLights.begin(), modelLights.end());
				for(; lightSize < lights.size(); lightSize++)
				{
					LightTriangle& tri = lights[lightSize];
					tri.V0 = make_float3(transform(trans, make_float4(tri.V0, 1)));
					tri.V1 = make_float3(transform(trans, make_float4(tri.V1, 1)));
					tri.V2 = make_float3(transform(trans, make_float4(tri.V2, 1)));
					tri.N  = normalize(make_float3(transform(trans, make_float4(tri.N, 0))));
					tri.instIx = static_cast<int32_t>(i);
				}
			}
		}
		return lights;
	}
}
