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
	Model::Model(const std::string& filePath) :
		Resource(filePath)
	{
	}



	void Model::AddMesh(const std::vector<float3>& vertices, const std::vector<float3>& normals, const std::vector<float2>& texCoords,
						const std::vector<uint3>& indices, uint32_t materialIndex)
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
		mNormalBuffer.Upload(mNormals, true);
		mTexcoordBuffer.Upload(mTexCoords, true);
		mIndexBuffer.Upload(mIndices, true);
		mMaterialIndexBuffer.Upload(mMaterialIndices, true);

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

		// CUDA
		mCudaMesh.vertices   = mVertexBuffer.Ptr<float3>();
		mCudaMesh.normals    = mNormalBuffer.Ptr<float3>();
		mCudaMesh.texcoords  = mTexcoordBuffer.Ptr<float2>();
		mCudaMesh.indices    = mIndexBuffer.Ptr<uint3>();
		mCudaMesh.matIndices = mMaterialIndexBuffer.Ptr<uint32_t>();

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

		mAccelBuffer.Alloc(compactedSize);
		OPTIX_CHECK(optixAccelCompact(optixContext, stream, mTraversableHandle, mAccelBuffer.DevicePtr(), mAccelBuffer.Size(), &mTraversableHandle));
		CUDA_CHECK(cudaDeviceSynchronize());
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
