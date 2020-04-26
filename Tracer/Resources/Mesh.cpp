#include "Resources/Mesh.h"

namespace Tracer
{
	Mesh::Mesh(const std::string& name,
			   const std::vector<float3>& vertices,
			   const std::vector<float3>& normals,
			   const std::vector<float2>& texCoords,
			   const std::vector<uint3>& indices,
			   std::shared_ptr<Material> material) :
		mName(name),
		mVertices(vertices),
		mNormals(normals),
		mTexCoords(texCoords),
		mIndices(indices),
		mMaterial(material)
	{
	}
}
