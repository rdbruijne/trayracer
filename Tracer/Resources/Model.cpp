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



	void Model::AddMesh(const std::vector<float3>& vertices, const std::vector<float3>& normals,
						const std::vector<float2>& texCoords, const std::vector<uint3>& indices,
						uint32_t materialIndex)
	{
		const uint32_t indexOffset = static_cast<uint32_t>(mVertices.size());

		mVertices.insert(mVertices.end(), vertices.begin(), vertices.end());
		mNormals.insert(mNormals.end(), normals.begin(), normals.end());
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

			// vertex positions
			t.v0 = mVertices[ix.x];
			t.v1 = mVertices[ix.y];
			t.v2 = mVertices[ix.z];

			// texcoords
			t.uv0x = mTexCoords[ix.x].x;
			t.uv0y = mTexCoords[ix.x].y;

			t.uv1x = mTexCoords[ix.y].x;
			t.uv1y = mTexCoords[ix.y].y;

			t.uv2x = mTexCoords[ix.z].x;
			t.uv2y = mTexCoords[ix.z].y;

			// normals
			t.N0 = mNormals[ix.x];
			t.N1 = mNormals[ix.y];
			t.N2 = mNormals[ix.z];

			t.N = normalize(t.N0 + t.N1 + t.N2);

			// mat ix
			t.matIx = mMaterialIndices[i];

			// add to the vector
			mPackedTriangles.push_back(t);
		}

		// CUDA
		mTriangleBuffer.Upload(mPackedTriangles, true);
		mCudaMesh.triangles  = mTriangleBuffer.Ptr<PackedTriangle>();

		// Acceleration setup
		OptixAccelBuildOptions buildOptions = {};
		buildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
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



	bool Model::BuildLights()
	{

		// check for emissive changes
		bool emissiveChanged = false;
		bool hasEmissiveMaterial = false;
		for(auto& mat : mMaterials)
		{
			if(mat->EmissiveChanged())
			{
				emissiveChanged = true;
				break;
			}
			const float3& em = mat->Emissive();
			if(em.x + em.y + em.z > Epsilon)
				hasEmissiveMaterial = true;
		}

		// determine is a light build is required
		if(!emissiveChanged && (!hasEmissiveMaterial || !IsDirty(false)))
			return false;

		// allocate memory
		size_t oldLightTriSize = mLightTriangles.size();
		mLightTriangles.clear();
		mLightTriangles.reserve(oldLightTriSize);

		// fill buffer
		for(size_t i = 0; i < mIndices.size(); i++)
		{
			const uint32_t matIx = mMaterialIndices[i];
			auto& mat = mMaterials[matIx];

			const float3& em = mat->Emissive();
			if(em.x + em.y + em.z > Epsilon)
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

		// clear unused memory
		mLightTriangles.shrink_to_fit();
		return true;
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
