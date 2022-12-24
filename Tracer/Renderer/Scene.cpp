#include "Scene.h"

// Project
#include "Renderer/Sky.h"
#include "Resources/Model.h"
#include "Resources/Instance.h"
#include "Resources/Material.h"

// C++
#include <map>
#include <set>
#include <string>

namespace Tracer
{

	Scene::Scene() :
		mSky(std::make_shared<Sky>())
	{

	}



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

		if(mSky->IsDirty())
			return true;

		for(const std::shared_ptr<Instance>& i : mInstances)
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
		for(const std::shared_ptr<Model>& m : mModels)
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
		for(const std::shared_ptr<Instance>& i : mInstances)
			models.insert(i->GetModel());
		return models.size();
	}



	size_t Scene::TextureCount() const
	{
		std::set<std::shared_ptr<Texture>> textures;
		for(const std::shared_ptr<Model>& mdl : mModels)
		{
			for(const std::shared_ptr<Material>& mat : mdl->Materials())
			{
				for(size_t i = 0; i < magic_enum::enum_count<MaterialPropertyIds>(); i++)
				{
					const std::shared_ptr<Texture>& tex = mat->TextureMap(static_cast<MaterialPropertyIds>(i));
					if(tex)
						textures.insert(tex);
				}
			}
		}
		return textures.size();
	}



	size_t Scene::TriangleCount() const
	{
		size_t triCount = 0;
		for(const std::shared_ptr<Instance>& i : mInstances)
			triCount += i->GetModel()->PolyCount();
		return triCount;
	}



	size_t Scene::UniqueTriangleCount() const
	{
		std::set<std::shared_ptr<Model>> models;
		for(const std::shared_ptr<Instance>& i : mInstances)
			models.insert(i->GetModel());

		size_t triCount = 0;
		for(const std::shared_ptr<Model>& m : models)
			triCount += m->PolyCount();
		return triCount;
	}



	size_t Scene::LightCount() const
	{
		size_t lightCount = 0;
		for(const std::shared_ptr<Instance>& i : mInstances)
			lightCount += i->GetModel()->LightCount();
		return lightCount;
	}



	size_t Scene::UniqueLightCount() const
	{
		std::set<std::shared_ptr<Model>> models;
		for(const std::shared_ptr<Instance>& i : mInstances)
			models.insert(i->GetModel());

		size_t lightCount = 0;
		for(const std::shared_ptr<Model>& m : models)
			lightCount += m->LightCount();
		return lightCount;
	}



	float Scene::LightEnergy() const
	{
		const std::vector<LightTriangle>& lightData = Lights();
		return lightData.size() == 0 ? 0 : lightData.back().sumEnergy;
	}



	float Scene::SunEnergy() const
	{
		return mSky->SunEnergy();
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

		Add(instance->GetModel());
		if(instance && std::find(mInstances.begin(), mInstances.end(), instance) == mInstances.end())
		{
			mInstances.push_back(instance);
			MarkDirty();
		}
	}



	void Scene::Remove(std::shared_ptr<Model> model)
	{
		std::vector<std::shared_ptr<Model>>::const_iterator it = std::find(mModels.begin(), mModels.end(), model);
		if(it != mModels.end())
		{
			bool removed;
			do
			{
				removed = false;
				for(const std::shared_ptr<Instance>& i : mInstances)
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
		std::vector<std::shared_ptr<Instance>>::const_iterator it = std::find(mInstances.begin(), mInstances.end(), instance);
		if(it != mInstances.end())
		{
			mInstances.erase(it);
			MarkDirty();
		}
	}



	std::shared_ptr<Tracer::Model> Scene::GetModel(const std::string& name) const
	{
		for(const std::shared_ptr<Model>& m : mModels)
		{
			if(m->Name() == name)
				return m;
		}
		return nullptr;
	}



	std::shared_ptr<Tracer::Material> Scene::GetMaterial(uint32_t instanceIx, uint32_t primIx) const
	{
		if(instanceIx >= mInstances.size())
			return nullptr;
		return mInstances[instanceIx]->GetModel()->GetMaterial(primIx);
	}



	std::shared_ptr<Tracer::Texture> Scene::GetTexture(const std::string& path) const
	{
		for(const std::shared_ptr<Model>& mdl : mModels)
		{
			for(const std::shared_ptr<Material>& mat : mdl->Materials())
			{
				for(size_t i = 0; i < magic_enum::enum_count<MaterialPropertyIds>(); i++)
				{
					const std::shared_ptr<Texture>& tex = mat->TextureMap(static_cast<MaterialPropertyIds>(i));
					if(tex && tex->Path() == path)
						return tex;
				}
			}
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
			const std::shared_ptr<Instance>& inst = mInstances[i];
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
	}
}
