#pragma once

// Project
#include "CUDA/CudaBuffer.h"
#include "Optix/Optix7.h"
#include "Resources/Material.h"
#include "Resources/Mesh.h"

// C++
#include <memory>

namespace Tracer
{
	class Renderer;
	class Scene
	{
		friend class Renderer;
	public:
		void AddMaterial(std::shared_ptr<Material> material);
		void AddMesh(std::shared_ptr<Mesh> mesh);

		inline bool IsDirty() const { return mIsDirty; }
		inline void ResetDirtyFlag() { mIsDirty = false; }

	private:
		// utility
		bool mIsDirty = false;

		// resources
		std::vector<std::shared_ptr<Material>> mMaterials;
		std::vector<std::shared_ptr<Mesh>> mMeshes;
	};
}
