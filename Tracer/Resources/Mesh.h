#pragma once

// Project
#include "Utility/LinearMath.h"

// C++
#include <memory>
#include <string>
#include <vector>

namespace Tracer
{
	class Material;
	class Mesh
	{
	public:
		explicit Mesh(const std::string& name,
					  const std::vector<float3>& vertices,
					  const std::vector<float3>& normals,
					  const std::vector<float3>& texCoords,
					  const std::vector<uint3>& indices,
					  std::shared_ptr<Material> material);

		inline std::shared_ptr<Material> Mat() { return mMaterial; }
		inline const std::vector<float3>& Vertices() const { return mVertices; }
		inline const std::vector<float3>& Normals() const { return mNormals; }
		inline const std::vector<float3>& Texcoords() const { return mTexCoords; }
		inline const std::vector<uint3>& Indices() const { return mIndices; }

	private:
		std::string mName = "";
		std::vector<float3> mVertices;
		std::vector<float3> mNormals;
		std::vector<float3> mTexCoords;
		std::vector<uint3>  mIndices;
		std::shared_ptr<Material> mMaterial;
	};
}
