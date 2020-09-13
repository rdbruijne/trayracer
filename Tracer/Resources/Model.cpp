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



	void Model::Set(const std::vector<float3>& vertices, const std::vector<float3>& normals, const std::vector<float2>& texCoords,
					const std::vector<uint3>& indices, std::vector<uint32_t> materialIndices)
	{
		mVertices = vertices;
		mNormals = normals;
		mTexCoords = texCoords;
		mIndices = indices;
		mMaterialIndices = materialIndices;
		MarkDirty();
	}



	void Model::Build(OptixDeviceContext optixContext, CUstream stream)
	{
		// make packed triangles
		mPackedTriangles.reserve(mIndices.size());
		for(size_t i = 0; i < mIndices.size(); i++)
		{
			const uint3 ix = mIndices[i];
			PackedTriangle t = {};

			// vertex positions
			const float3& v0 = mVertices[ix.x];
			t.v0x = v0.x;
			t.v0y = v0.y;
			t.v0z = v0.z;

			const float3& v1 = mVertices[ix.y];
			t.v1x = v1.x;
			t.v1y = v1.y;
			t.v1z = v1.z;

			const float3& v2 = mVertices[ix.z];
			t.v2x = v2.x;
			t.v2y = v2.y;
			t.v2z = v2.z;

			// texcoords
			const float2& uv0 = mTexCoords[ix.x];
			t.uv0x = __float2half(uv0.x);
			t.uv0y = __float2half(uv0.y);

			const float2& uv1 = mTexCoords[ix.y];
			t.uv1x = __float2half(uv1.x);
			t.uv1y = __float2half(uv1.y);

			const float2& uv2 = mTexCoords[ix.z];
			t.uv2x = __float2half(uv2.x);
			t.uv2y = __float2half(uv2.y);

			// normals
			const float3& N0 = mNormals[ix.x];
			t.N0x = __float2half(N0.x);
			t.N0y = __float2half(N0.y);
			t.N0z = __float2half(N0.z);

			const float3& N1 = mNormals[ix.y];
			t.N1x = __float2half(N1.x);
			t.N1y = __float2half(N1.y);
			t.N1z = __float2half(N1.z);

			const float3& N2 = mNormals[ix.z];
			t.N2x = __float2half(N2.x);
			t.N2y = __float2half(N2.y);
			t.N2z = __float2half(N2.z);

			const float3 N = normalize(N0 + N1 + N2);
			t.Nx = __float2half(N.x);
			t.Ny = __float2half(N.y);
			t.Nz = __float2half(N.z);

			// add to the vector
			mPackedTriangles.push_back(t);
		}

		// CUDA
		mTriangleBuffer.Upload(mPackedTriangles, true);
		mMaterialIndexBuffer.Upload(mMaterialIndices, true);
		mCudaMesh.triangles = mTriangleBuffer.Ptr<PackedTriangle>();
		mCudaMesh.materialIndices = mMaterialIndexBuffer.Ptr<uint32_t>();

		// prepare build input
		mBuildInput = {};
		mBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		// vertices
		mBuildInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
		mBuildInput.triangleArray.vertexStrideInBytes = sizeof(PackedTriangle) / 3;
		mBuildInput.triangleArray.numVertices         = static_cast<unsigned int>(mPackedTriangles.size() * 3);
		mBuildInput.triangleArray.vertexBuffers       = mTriangleBuffer.DevicePtrPtr();

		// other
		static uint32_t buildFlags[] = { 0 };
		mBuildInput.triangleArray.flags              = buildFlags;
		mBuildInput.triangleArray.numSbtRecords      = 1;

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
