#include "Model.h"

// project
#include "Renderer/OptixError.h"
#include "Resources/Material.h"

// Optix
#pragma warning(push)
#pragma warning(disable: 4061)
#include "optix7/optix.h"
#include "optix7/optix_stubs.h"
#pragma warning(pop)

namespace Tracer
{
	Model::Model(const std::string& filePath, const std::string& name) :
		Resource(name.empty() ? FileName(filePath) : name),
		mFilePath(filePath)
	{
	}



	std::shared_ptr<Tracer::Material> Model::GetMaterial(uint32_t primIx) const
	{
		if(primIx > mMaterialIndices.size())
			return nullptr;
		return mMaterials[mMaterialIndices[primIx]];
	}



	std::shared_ptr<Tracer::Material> Model::GetMaterial(const std::string& name) const
	{
		for(auto m : mMaterials)
		{
			if(m->Name() == name)
				return m;
		}
		return nullptr;
	}



	uint32_t Model::AddMaterial(std::shared_ptr<Material> mat)
	{
		mMaterials.push_back(mat);
		AddDependency(mat);
		MarkDirty();
		return static_cast<uint32_t>(mMaterials.size() - 1);
	}



	void Model::AddMesh(const std::vector<float3>& vertices, const std::vector<float3>& normals, const std::vector<float3>& tangents,
					 const std::vector<float3>& bitangents, const std::vector<float2>& texCoords, const std::vector<uint3>& indices,
					 uint32_t materialIndex)
	{
		const uint32_t indexOffset = static_cast<uint32_t>(mVertices.size());

		mVertices.insert(mVertices.end(), vertices.begin(), vertices.end());
		mNormals.insert(mNormals.end(), normals.begin(), normals.end());
		mTangents.insert(mTangents.end(), tangents.begin(), tangents.end());
		mBitangents.insert(mBitangents.end(), bitangents.begin(), bitangents.end());
		mTexCoords.insert(mTexCoords.end(), texCoords.begin(), texCoords.end());

		mIndices.reserve(mIndices.size() + indices.size());
		mMaterialIndices.reserve(mMaterialIndices.size() + indices.size());
		for(const auto& i : indices)
		{
			mIndices.push_back(i + indexOffset);
			mMaterialIndices.push_back(materialIndex);
		}

		MarkDirty();
	}



	void Model::Build(OptixDeviceContext optixContext, CUstream stream)
	{
		// upload buffers
		mVertexBuffer.Upload(mVertices, true);
		mIndexBuffer.Upload(mIndices, true);

		// prepare build input
		mBuildInput = {};
		mBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		// vertices
		mBuildInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3; // #TODO: float4 for better cache alignment?
		mBuildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
		mBuildInput.triangleArray.numVertices         = static_cast<unsigned int>(mVertices.size());
		mBuildInput.triangleArray.vertexBuffers       = mVertexBuffer.DevicePtrPtr();

		// indices
		mBuildInput.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		mBuildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
		mBuildInput.triangleArray.numIndexTriplets   = static_cast<unsigned int>(mIndices.size());
		mBuildInput.triangleArray.indexBuffer        = mIndexBuffer.DevicePtr();

		// other
		static uint32_t buildFlags[] = { 0 };
		mBuildInput.triangleArray.flags              = buildFlags;
		mBuildInput.triangleArray.numSbtRecords      = 1;

		// make packed triangles
		mPackedTriangles.reserve(mIndices.size());
		for(size_t i = 0; i < mIndices.size(); i++)
		{
			const uint3 ix = mIndices[i];
			PackedTriangle t = {};

			t.uv0 = mTexCoords[ix.x];
			t.uv1 = mTexCoords[ix.y];
			t.uv2 = mTexCoords[ix.z];

			t.matIx = mMaterialIndices[i];

			t.N0 = mNormals[ix.x];
			t.N1 = mNormals[ix.y];
			t.N2 = mNormals[ix.z];

			const float3 v0 = mVertices[ix.x];
			const float3 v1 = mVertices[ix.y];
			const float3 v2 = mVertices[ix.z];

#if false
			float3 N = normalize(cross(v1 - v0, v2 - v0));
			t.Nx = N.x;
			t.Ny = N.y;
			t.Nz = N.z;
#else
			float3 N = normalize(t.N0 + t.N1 + t.N2);
			t.Nx = N.x;
			t.Ny = N.y;
			t.Nz = N.z;
#endif

			t.tangent = mTangents[ix.x];
			t.bitangent = mBitangents[ix.x];

			mPackedTriangles.push_back(t);
		}

		// CUDA
		mTriangleBuffer.Upload(mPackedTriangles, true);
		mCudaMesh.triangles  = mTriangleBuffer.Ptr<PackedTriangle>();

		// Acceleration setup
		OptixAccelBuildOptions buildOptions = {};
		buildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;// | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
		buildOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

		OptixAccelBufferSizes accelBufferSizes = {};
		OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &buildOptions, &mBuildInput, 1, &accelBufferSizes));

		// Prepare for compacting
		CudaBuffer compactedSizeBuffer(sizeof(uint64_t));

		OptixAccelEmitDesc emitDesc;
		emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = compactedSizeBuffer.DevicePtr();

		// Execute build
		CudaBuffer tempBuffer(accelBufferSizes.tempSizeInBytes);
		CudaBuffer outputBuffer(accelBufferSizes.outputSizeInBytes);
		OPTIX_CHECK(optixAccelBuild(optixContext, stream, &buildOptions, &mBuildInput, 1, tempBuffer.DevicePtr(), tempBuffer.Size(),
									outputBuffer.DevicePtr(), outputBuffer.Size(), &mTraversableHandle, &emitDesc, 1));

		// Compact
		uint64_t compactedSize = 0;
		compactedSizeBuffer.Download(&compactedSize);

		if(mAccelBuffer.Size() != compactedSize)
			mAccelBuffer.Alloc(compactedSize);
		OPTIX_CHECK(optixAccelCompact(optixContext, stream, mTraversableHandle, mAccelBuffer.DevicePtr(), mAccelBuffer.Size(), &mTraversableHandle));
		CUDA_CHECK(cudaDeviceSynchronize());
	}



	void Model::BuildLights()
	{
		size_t oldLightTriSize = mLightTriangles.size();
		mLightTriangles.clear();
		mLightTriangles.reserve(oldLightTriSize);

		for(size_t i = 0; i < mIndices.size(); i++)
		{
			const uint32_t matIx = mMaterialIndices[i];
			auto& mat = mMaterials[matIx];
			const float3& em = mat->Emissive();
			if(em.x + em.y + em.z > 0)
			{
				LightTriangle lt = {};

				lt.triIx = static_cast<int32_t>(i);

				const uint3& indices = mIndices[i];
				lt.V0 = mVertices[indices.x];
				lt.V1 = mVertices[indices.y];
				lt.V2 = mVertices[indices.z];
				lt.N = normalize(cross(lt.V1 - lt.V0, lt.V2 - lt.V0));

				const float a = length(lt.V1 - lt.V0);
				const float b = length(lt.V2 - lt.V1);
				const float c = length(lt.V0 - lt.V2);
				const float s = (a + b + c) * .5f;
				lt.area = sqrtf(s * (s-a) * (s-b) * (s-c));

				lt.radiance = em;

				const float3 energy = em * lt.area;
				lt.energy = energy.x + energy.y + energy.z;

				mLightTriangles.push_back(lt);
			}
		}

		mLightTriangles.shrink_to_fit();
	}



	OptixInstance Model::InstanceData(uint32_t instanceId, const float3x4& transform) const
	{
		OptixInstance inst = {};
		memcpy(inst.transform, &transform, 12 * sizeof(float));
		inst.instanceId        = instanceId;
		inst.sbtOffset         = 0;
		inst.visibilityMask    = 0xFF;
		inst.flags             = OPTIX_INSTANCE_FLAG_NONE | OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM;
		inst.traversableHandle = mTraversableHandle;
		return inst;
	}
}
