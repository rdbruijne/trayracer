#include "Mesh.h"

namespace Tracer
{

	Mesh::Mesh(const std::vector<float3>& vertices, const std::vector<float3>& normals,
			   const std::vector<float3>& texCoords, const std::vector<uint3>& indices) :
		mVertices(vertices),
		mNormals(normals),
		mTexCoords(texCoords),
		mIndices(indices)
	{
	}
}
