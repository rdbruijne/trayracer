#pragma once

// Project
#include "Common/CommonStructs.h"
#include "Renderer/CudaBuffer.h"
#include "Resources/Resource.h"
#include "Utility/LinearMath.h"

// Optix
#include "optix7/optix.h"

// C++
#include <memory>
#include <string>
#include <vector>

namespace Tracer
{
	class Material;
	class Model : public Resource
	{
	public:
		Model() = default;
		explicit Model(const std::string& name);

		// materials
		const std::vector<std::shared_ptr<Material>>& Materials() const { return mMaterials; }
		uint32_t AddMaterial(std::shared_ptr<Material> mat);

		// meshes
		void AddMesh(const std::vector<float3>& vertices, const std::vector<float3>& normals, const std::vector<float2>& texCoords,
					 const std::vector<uint3>& indices, uint32_t materialIndex);

		// build
		void Build(OptixDeviceContext optixContext, CUstream stream);

		// build info
		OptixInstance InstanceData(uint32_t instanceId, const float3x4& transform) const;
		inline CudaMeshData CudaMesh() { return mCudaMesh; }
		inline const CudaMeshData& CudaMesh() const { return mCudaMesh; }

	private:
		std::vector<std::shared_ptr<Material>> mMaterials;

		// geometry
		std::vector<float3> mVertices;
		std::vector<float3> mNormals;
		std::vector<float2> mTexCoords;
		std::vector<uint3>  mIndices;
		std::vector<uint32_t> mMaterialIndices;

		// build data
		CudaBuffer mVertexBuffer = {};
		CudaBuffer mNormalBuffer = {};
		CudaBuffer mTexcoordBuffer = {};
		CudaBuffer mIndexBuffer = {};
		CudaBuffer mMaterialIndexBuffer = {};

		OptixBuildInput mBuildInput = {};
		OptixTraversableHandle mTraversableHandle = 0;
		CudaBuffer mAccelBuffer = {};

		CudaMeshData mCudaMesh = {};
	};
}
