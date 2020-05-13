#pragma once

// C++
#include <memory>
#include <string>
#include <vector>

namespace Tracer
{
	class CameraNode;
	class Instance;
	class Material;
	class Model;
	class Renderer;
	class Scene
	{
		friend class Renderer;
	public:
		void Load(const std::string& sceneFile, CameraNode& camNode);

		bool IsDirty() const;
		inline void MarkClean() { mIsDirty = false; }
		inline void MarkDirty() { mIsDirty = true; }

		size_t InstanceCount() const;
		size_t MaterialCount() const;
		size_t ModelCount() const;
		size_t InstancedModelCount() const;
		size_t TriangleCount() const;
		size_t UniqueTriangleCount() const;

		void AddModel(std::shared_ptr<Model> model);
		void AddInstance(std::shared_ptr<Instance> instance);

		std::shared_ptr<Material> GetMaterial(uint32_t instanceIx, uint32_t primIx);

		inline const std::vector<std::shared_ptr<Model>>& Models() const { return mModels; }
		inline const std::vector<std::shared_ptr<Instance>>& Instances() const { return mInstances; }

	private:
		// utility
		bool mIsDirty = false;

		// resources
		std::vector<std::shared_ptr<Model>> mModels;
		std::vector<std::shared_ptr<Instance>> mInstances;
	};
}
