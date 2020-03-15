#pragma once

// CUDA
#include <vector_types.h>

// C++
#include <vector>

namespace Tracer
{
	class Mesh
	{
	public:
		explicit Mesh(const std::vector<float3>& vertices, const std::vector<float3>& normals,
					  const std::vector<float3>& texCoords, const std::vector<uint3>& indices);

		inline const std::vector<float3>& GetVertices() const
		{
			return mVertices;
		}

		inline const std::vector<float3>& GetNormals() const
		{
			return mNormals;
		}

		inline const std::vector<float3>& GetTexcoords() const
		{
			return mTexCoords;
		}

		inline const std::vector<uint3>& GetIndices() const
		{
			return mIndices;
		}

	private:
		std::vector<float3> mVertices;
		std::vector<float3> mNormals;
		std::vector<float3> mTexCoords;
		std::vector<uint3>  mIndices;
	};
}
