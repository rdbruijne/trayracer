#include "Instance.h"

// Project
#include "Resources/Model.h"

namespace Tracer
{
	Instance::Instance(const std::string& name, std::shared_ptr<Model> model, const float3x4& transform) :
		Resource(name),
		mTransform(transform),
		mModel(model)
	{
		AddDependency(model);
		decompose(mTransform, mPos, mEuler, mScale);
	}



	void Instance::SetTransform(const float3x4& transform)
	{
		mTransform = transform;
		decompose(mTransform, mPos, mEuler, mScale);
		MarkDirty();
	}



	void Instance::SetDecomposedTransform(const float3& pos, const float3& euler, const float3& scale)
	{
		mPos = pos;
		mEuler = euler;
		mScale = scale;
		mTransform = rotate_3x4(euler) * scale_3x4(scale) * translate_3x4(pos);
		MarkDirty();
	}



	OptixInstance Instance::InstanceData(uint32_t instanceId) const
	{
		OptixInstance inst = {};
		memcpy(inst.transform, &mTransform, 12 * sizeof(float));
		inst.instanceId        = instanceId;
		inst.sbtOffset         = 0;
		inst.visibilityMask    = mVisible ? 0xFFu : 0x00u;
		inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
		inst.traversableHandle = mModel->TraversableHandle();
		return inst;
	}
}
