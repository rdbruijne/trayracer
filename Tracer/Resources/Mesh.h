#pragma once

// CUDA
#include <vector_types.h>

// C++
#include <vector>

namespace Tracer
{
	/*!
	 * @brief Mesh data.
	 */
	class Mesh
	{
	public:
		/*!
		 * @brief Construct a mesh.
		 * @param[in] vertices Vertex positions.
		 * @param[in] normals Vertex normals.
		 * @param[in] texCoords Vertex texture coordinates.
		 * @param[in] indices Vertex indices.
		 */
		explicit Mesh(const std::vector<float3>& vertices, const std::vector<float3>& normals,
					  const std::vector<float3>& texCoords, const std::vector<uint3>& indices);

		/*!
		 * @brief Get the position data.
		 * @return Reference to the position data.
		 */
		const std::vector<float3>& GetVertices() const
		{
			return mVertices;
		}

		/*!
		 * @brief Get the normal data.
		 * @return Reference to the normal data.
		 */
		const std::vector<float3>& GetNormals() const
		{
			return mNormals;
		}

		/*!
		 * @brief Get the texture coordinate data.
		 * @return Reference to the texture coordinate data.
		 */
		const std::vector<float3>& GetTexcoords() const
		{
			return mTexCoords;
		}

		/*!
		 * @brief Get the index data.
		 * @return Reference to the index data.
		 */
		const std::vector<uint3>& GetIndices() const
		{
			return mIndices;
		}

	private:
		/*! @{ Vertex positions. */
		std::vector<float3> mVertices;
		/*! Vertex normals. */
		std::vector<float3> mNormals;
		/*! Vertex texture coordinates. */
		std::vector<float3> mTexCoords;
		/*! Vertex indices. */
		std::vector<uint3>  mIndices;
		/*! @} */
	};
}
