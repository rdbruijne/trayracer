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
	}
}
