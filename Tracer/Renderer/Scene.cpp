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



	size_t Scene::TextureCount() const
	{
		return mTextures.size();
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



	void Scene::Add(std::shared_ptr<Model> model)
	{
		if(model && std::find(mModels.begin(), mModels.end(), model) == mModels.end())
		{
			mModels.push_back(model);
			MarkDirty();
		}
	}



	void Scene::Add(std::shared_ptr<Instance> instance)
	{
		assert(instance->GetModel() != nullptr);
		assert(std::find(mModels.begin(), mModels.end(), instance->GetModel()) != mModels.end());

		if(instance && std::find(mInstances.begin(), mInstances.end(), instance) == mInstances.end())
		{
			mInstances.push_back(instance);
			MarkDirty();
		}
	}



	void Scene::Remove(std::shared_ptr<Model> model)
	{
		auto it = std::find(mModels.begin(), mModels.end(), model);
		if(it != mModels.end())
		{
			bool removed;
			do
			{
				removed = false;
				for(auto i : mInstances)
				{
					if(i->GetModel() == model)
					{
						Remove(i);
						removed = true;
						break;
					}
				}
			} while(removed);
			mModels.erase(it);
			MarkDirty();
		}
	}



	void Scene::Remove(std::shared_ptr<Instance> instance)
	{
		auto it = std::find(mInstances.begin(), mInstances.end(), instance);
		if(it != mInstances.end())
		{
			mInstances.erase(it);
			MarkDirty();
		}
	}



	std::shared_ptr<Tracer::Model> Scene::GetModel(const std::string& name) const
	{
		for(auto& m : mModels)
		{
			if(m->Name() == name)
				return m;
		}
		return nullptr;
	}



	std::shared_ptr<Tracer::Material> Scene::GetMaterial(uint32_t instanceIx, uint32_t primIx)
	{
		if(instanceIx >= mInstances.size())
			return nullptr;
		return mInstances[instanceIx]->GetModel()->GetMaterial(primIx);
	}



	std::shared_ptr<Tracer::Texture> Scene::GetTexture(const std::string& path)
	{
		for(auto& t : mTextures)
		{
			if(t->Path() == path)
				return t;
		}
		return nullptr;
	}



	void Scene::GatherLights()
	{
		mLights.clear();
		size_t lightSize = 0;
		float sumEnergy = 0;
		for(size_t i = 0; i < mInstances.size(); i++)
		{
			auto inst = mInstances[i];
			const std::vector<LightTriangle>& modelLights = inst->GetModel()->Lights();
			const float3x4& trans = inst->Transform();
			if(modelLights.size() > 0)
			{
				mLights.insert(mLights.end(), modelLights.begin(), modelLights.end());
				for(; lightSize < mLights.size(); lightSize++)
				{
					LightTriangle& tri = mLights[lightSize];
					tri.V0 = make_float3(transform(trans, make_float4(tri.V0, 1)));
					tri.V1 = make_float3(transform(trans, make_float4(tri.V1, 1)));
					tri.V2 = make_float3(transform(trans, make_float4(tri.V2, 1)));
					tri.N  = normalize(make_float3(transform(trans, make_float4(tri.N, 0))));
					tri.sumEnergy = sumEnergy;
					tri.instIx = static_cast<int32_t>(i);

					sumEnergy += tri.energy;
				}
			}
		}

		// set pdf
		// #TODO: pdf
		/*for(size_t i = 0; i < mLights.size(); i++)
		{
			LightTriangle& tri = mLights[i];

			static const float3 luminanceFactor = make_float3(0.2126f, 0.7152f, 0.0722f);
			const float3 luminance = tri.radiance * luminanceFactor;
			const float3 normal = cross(tri.V1 - tri.V0, tri.V2 - tri.V0);
			const float importance = (luminance.x + luminance.y + luminance.z) * length(normal) * 0.5f;
		}*/
	}
}
