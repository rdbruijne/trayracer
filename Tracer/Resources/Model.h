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
		explicit Model(const std::string& filePath, const std::string& name = "");

		// info
		const std::string& FilePath() const { return mFilePath; }
		inline size_t LightCount() const { return mLightTriangles.size(); }
		inline size_t PolyCount() const { return mIndices.size(); }
		inline size_t VertexCount() const { return mVertices.size(); }

		// materials
		std::shared_ptr<Material> GetMaterial(uint32_t primIx) const;
		std::shared_ptr<Material> GetMaterial(const std::string& name) const;
		const std::vector<std::shared_ptr<Material>>& Materials() const { return mMaterials; }
		uint32_t AddMaterial(std::shared_ptr<Material> mat);

		// meshes
		void AddMesh(const std::vector<float3>& vertices, const std::vector<float3>& normals,
					 const std::vector<float2>& texCoords, const std::vector<uint3>& indices,
					 uint32_t materialIndex);

		// build
		void Build(OptixDeviceContext optixContext, CUstream stream);
		bool BuildLights();

		// build info
		OptixInstance InstanceData(uint32_t instanceId, const float3x4& transform) const;
		inline CudaMeshData CudaMesh() { return mCudaMesh; }
		inline const CudaMeshData& CudaMesh() const { return mCudaMesh; }
		inline const std::vector<LightTriangle>& Lights() const { return mLightTriangles; }

	private:
		std::string mFilePath;
		std::vector<std::shared_ptr<Material>> mMaterials;

		// geometry
		std::vector<float3> mVertices;
		std::vector<float3> mNormals;
		std::vector<float2> mTexCoords;
		std::vector<uint3>  mIndices;
		std::vector<uint32_t> mMaterialIndices;
		std::vector<PackedTriangle> mPackedTriangles;
		std::vector<LightTriangle> mLightTriangles;

		// build data
		CudaBuffer mTriangleBuffer = {};
		CudaBuffer mMaterialIndexBuffer = {};

		OptixBuildInput mBuildInput = {};
		OptixTraversableHandle mTraversableHandle = 0;
		CudaBuffer mAccelBuffer = {};

		CudaMeshData mCudaMesh = {};
	};
}
