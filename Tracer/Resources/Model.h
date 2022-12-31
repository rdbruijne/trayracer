#pragma once

// Project
#include "Common/CommonStructs.h"
#include "CUDA/CudaBuffer.h"
#include "Resources/Resource.h"
#include "Utility/LinearMath.h"

// Optix
#include "optix7/optix.h"

// C++
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Tracer
{
	class Material;
	class Renderer;
	class Model : public Resource
	{
	public:
		Model() = default;
		explicit Model(const std::string& filePath, const std::string& name = "");

		Model& operator =(const Model& m) = delete;

		// info
		const std::string& FilePath() const { return mFilePath; }
		inline size_t LightCount() const { return mLightTriangles.size(); }
		inline size_t PolyCount() const { return mIndices.size(); }
		inline size_t VertexCount() const { return mVertices.size(); }

		// geometry
		const std::vector<float3>& Vertices() const { return mVertices; }
		const std::vector<float3>& Normals() const { return mNormals; }
		const std::vector<float2>& TexCoords() const { return mTexCoords; }
		const std::vector<uint3>& Indices() const { return mIndices; }
		const std::vector<uint32_t>& MaterialIndices() const { return mMaterialIndices; }

		// materials
		std::shared_ptr<Material> GetMaterial(uint32_t primIx) const;
		std::shared_ptr<Material> GetMaterial(const std::string& name) const;
		const std::vector<std::shared_ptr<Material>>& Materials() const { return mMaterials; }
		uint32_t AddMaterial(std::shared_ptr<Material> mat);

		// meshes
		void AddMesh(const std::vector<float3>& vertices, const std::vector<float3>& normals,
					 const std::vector<float2>& texCoords, const std::vector<uint3>& indices,
					 uint32_t materialIndex);

		void Set(const std::vector<float3>& vertices, const std::vector<float3>& normals,
				 const std::vector<float2>& texCoords, const std::vector<uint3>& indices,
				 std::vector<uint32_t> materialIndices);

		// build
		void Build();
		bool BuildLights();

		// upload
		void Upload(Renderer* renderer);

		// build info
		inline CudaMeshData CudaMesh() { return mCudaMesh; }
		inline const CudaMeshData& CudaMesh() const { return mCudaMesh; }
		inline const std::vector<LightTriangle>& Lights() const { return mLightTriangles; }

		// optix info
		inline OptixTraversableHandle TraversableHandle() const { return mTraversableHandle; }

	private:
		std::string mFilePath;
		std::vector<std::shared_ptr<Material>> mMaterials;

		// geometry
		std::vector<float3> mVertices;
		std::vector<float3> mNormals;
		std::vector<float2> mTexCoords;
		std::vector<uint3>  mIndices;
		std::vector<uint32_t> mMaterialIndices;

		// mutex
		std::mutex mMutex;

		// build data
		std::vector<PackedTriangle> mPackedTriangles;
		std::vector<LightTriangle> mLightTriangles;

		// GPU data
		CudaBuffer mTriangleBuffer = {};
		CudaBuffer mMaterialIndexBuffer = {};

		OptixBuildInput mBuildInput = {};
		OptixTraversableHandle mTraversableHandle = 0;
		CudaBuffer mAccelBuffer = {};

		CudaMeshData mCudaMesh = {};
	};
}
