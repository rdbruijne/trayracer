#pragma once

// Project
#include "Common/CommonStructs.h"

// C++
#include <memory>
#include <string>
#include <vector>

namespace Tracer
{
	class Instance;
	class Material;
	class Model;
	class Sky;
	class Texture;
	class Window;
	class Scene
	{
	public:
		Scene();

		void Clear();

		bool IsDirty() const;
		inline void MarkClean() { mIsDirty = false; }
		inline void MarkDirty() { mIsDirty = true; }

		size_t InstanceCount() const;
		size_t MaterialCount() const;
		size_t ModelCount() const;
		size_t InstancedModelCount() const;
		size_t TextureCount() const;
		size_t TriangleCount() const;
		size_t UniqueTriangleCount() const;
		size_t LightCount() const;
		size_t UniqueLightCount() const;

		void Add(std::shared_ptr<Model> model);
		void Add(std::shared_ptr<Instance> instance);

		void Remove(std::shared_ptr<Model> model);
		void Remove(std::shared_ptr<Instance> instance);

		std::shared_ptr<Sky> GetSky() const { return mSky; }
		std::shared_ptr<Model> GetModel(const std::string& name) const;
		std::shared_ptr<Material> GetMaterial(uint32_t instanceIx, uint32_t primIx) const;
		std::shared_ptr<Texture> GetTexture(const std::string& path) const;

		inline const std::vector<std::shared_ptr<Model>>& Models() const { return mModels; }
		inline const std::vector<std::shared_ptr<Instance>>& Instances() const { return mInstances; }

		const std::vector<LightTriangle>& Lights() const { return mLights; }
		void GatherLights();

	private:
		// utility
		bool mIsDirty = false;

		// sky
		std::shared_ptr<Sky> mSky = nullptr;

		// resources
		std::vector<std::shared_ptr<Model>> mModels;
		std::vector<std::shared_ptr<Instance>> mInstances;

		// build data
		std::vector<LightTriangle> mLights;
	};
}
