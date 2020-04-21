#include "CameraNode.h"

namespace Tracer
{
	CameraNode::CameraNode(const float3& position, const float3& target, const float3& up, const float fov) :
		mPosition(position),
		mTarget(target),
		mUp(up),
		mFov(fov)
	{
	}

	void CameraNode::SetPosition(const float3& position)
	{
		if(mPosition != position)
		{
			mPosition = position;
			mHasChanged = true;
		}
	}

	void CameraNode::SetTarget(const float3& target)
	{
		if(mTarget != target)
		{
			mTarget = target;
			mHasChanged = true;
		}
	}

	void CameraNode::SetUp(const float3& up)
	{
		if(mUp != up)
		{
			mUp = up;
			mHasChanged = true;
		}
	}

	void CameraNode::SetFov(const float& fov)
	{
		if(mFov != fov)
		{
			mFov = fov;
			mHasChanged = true;
		}
	}
}
